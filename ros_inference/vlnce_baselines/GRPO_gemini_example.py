import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import copy

# --- 0. 占位符函数 (Assume they handle device placement internally if complex) ---
def placeholder_get_batch_prompts_initial_states(num_distinct_prompts, state_dim, device_for_initial_tensor='cpu'):
    # Returns on CPU by default, can be moved to GPU when first needed by a model
    return torch.randn(num_distinct_prompts, state_dim, device=device_for_initial_tensor)

def placeholder_batch_state_update(current_batch_states_chunk, batch_actions_chunk, action_dim, current_device):
    """
    Placeholder for state update. Assumes input chunks are on current_device.
    Returns next_batch_states_chunk on the same current_device.
    """
    # This function is critical. If it involves its own large network, that network also needs careful device management.
    # For this example, we assume it's a simple operation or its own model is on current_device.
    return torch.randn_like(current_batch_states_chunk) # Output on the same device as input

def placeholder_batch_reward_function(batch_responses_actions_tensor_cpu): # Expects CPU tensor
    num_responses = batch_responses_actions_tensor_cpu.shape[0]
    rewards = []
    for i in range(num_responses):
        actual_length = (batch_responses_actions_tensor_cpu[i] != 0).sum().item()
        rewards.append(actual_length * 0.1 + torch.randn(1).item() * 0.05)
    return torch.tensor(rewards, dtype=torch.float, device='cpu') # Return on CPU


# --- 1. 策略网络 (Actor) ---
class PolicyNetwork(nn.Module): # Unchanged
    def __init__(self, state_dim, action_dim, hidden_dim=256, dropout_rate=0.1):
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_actor = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_batch):
        x = F.relu(self.fc1(state_batch))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        action_logits = self.fc_actor(x)
        return action_logits

    def get_action_dist(self, state_batch):
        logits = self.forward(state_batch)
        return Categorical(logits=logits)

# --- 2. GRPO Agent ---
class GRPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 strict_model_batch_size,
                 sft_model_path=None,
                 lr_actor=1e-5,
                 eps_clip=0.2,
                 k_epochs=4,
                 beta_kl=0.05,
                 gamma_entropy=0.01,
                 device=torch.device("cpu")): # This 'device' is the primary compute device (GPU)

        self.device = device # Primary compute device (GPU)
        self.cpu_device = torch.device("cpu")
        self.strict_model_batch_size = strict_model_batch_size
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.beta_kl = beta_kl
        self.gamma_entropy = gamma_entropy
        self.action_dim = action_dim

        # Networks are on the primary compute device (GPU)
        self.actor_current = PolicyNetwork(state_dim, action_dim, dropout_rate=0.1).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor_current.parameters(), lr=lr_actor)
        self.actor_old = PolicyNetwork(state_dim, action_dim, dropout_rate=0.1).to(self.device)
        self.actor_reference = PolicyNetwork(state_dim, action_dim, dropout_rate=0.0).to(self.device)

        if sft_model_path: self.load_sft_weights(sft_model_path)
        else:
            self.actor_old.load_state_dict(self.actor_current.state_dict())
            self.actor_reference.load_state_dict(self.actor_current.state_dict())

        self.actor_reference.eval()
        for param in self.actor_reference.parameters(): param.requires_grad = False

    def load_sft_weights(self, sft_model_path):
        try:
            # Load to CPU first to potentially save GPU memory if model is large, then move
            sft_state_dict = torch.load(sft_model_path, map_location=self.cpu_device)
            self.actor_current.load_state_dict(sft_state_dict) # Loads into GPU model
            print(f"SFT weights loaded from: {sft_model_path} to device: {self.device}")
        except Exception as e:
            print(f"Error loading SFT weights: {e}. Using initial random weights for actor_current.")
        # Sync other models
        self.actor_old.load_state_dict(self.actor_current.state_dict())
        self.actor_reference.load_state_dict(self.actor_current.state_dict())


    def generate_responses_for_batch_prompts(self,
                                             batch_distinct_prompts_initial_states_cpu, # Expect CPU tensor
                                             num_samples_per_prompt,
                                             max_len_response):
        self.actor_old.eval() # actor_old is on self.device (GPU)
        num_distinct_prompts = batch_distinct_prompts_initial_states_cpu.shape[0]
        assert num_distinct_prompts <= self.strict_model_batch_size

        all_initial_states_collected_cpu = []
        all_actions_collected_cpu = []
        all_log_probs_collected_cpu = []
        all_prompt_indices_collected_cpu = []

        # Move initial prompts to GPU for this generation session
        batch_distinct_prompts_initial_states_gpu = batch_distinct_prompts_initial_states_cpu.to(self.device)

        for sample_iter in range(num_samples_per_prompt):
            current_physical_batch_states_gpu = batch_distinct_prompts_initial_states_gpu.clone().detach()
            
            current_pass_actions_over_time_gpu = []
            current_pass_log_probs_over_time_gpu = []

            for t in range(max_len_response):
                with torch.no_grad():
                    action_dist = self.actor_old.get_action_dist(current_physical_batch_states_gpu)
                    actions_at_t_gpu = action_dist.sample()
                    log_probs_at_t_gpu = action_dist.log_prob(actions_at_t_gpu)

                current_pass_actions_over_time_gpu.append(actions_at_t_gpu)
                current_pass_log_probs_over_time_gpu.append(log_probs_at_t_gpu)
                
                current_physical_batch_states_gpu = placeholder_batch_state_update(
                    current_physical_batch_states_gpu, actions_at_t_gpu, self.action_dim, self.device
                )
            
            pass_actions_tensor_gpu = torch.stack(current_pass_actions_over_time_gpu, dim=1)
            pass_log_probs_tensor_gpu = torch.stack(current_pass_log_probs_over_time_gpu, dim=1)

            # Move results for this pass to CPU for collection
            for i in range(num_distinct_prompts):
                all_initial_states_collected_cpu.append(batch_distinct_prompts_initial_states_cpu[i]) # Already on CPU
                all_actions_collected_cpu.append(pass_actions_tensor_gpu[i].to(self.cpu_device))
                all_log_probs_collected_cpu.append(pass_log_probs_tensor_gpu[i].to(self.cpu_device))
                all_prompt_indices_collected_cpu.append(i) # Scalar, effectively CPU

        del batch_distinct_prompts_initial_states_gpu # Free GPU memory
        del current_physical_batch_states_gpu, current_pass_actions_over_time_gpu, current_pass_log_probs_over_time_gpu
        del pass_actions_tensor_gpu, pass_log_probs_tensor_gpu
        if self.device.type == 'cuda': torch.cuda.empty_cache()


        final_initial_states_cpu = torch.stack(all_initial_states_collected_cpu)
        final_actions_tensor_cpu = torch.stack(all_actions_collected_cpu)
        final_log_probs_tensor_cpu = torch.stack(all_log_probs_collected_cpu)
        final_prompt_indices_cpu = torch.tensor(all_prompt_indices_collected_cpu, dtype=torch.long, device=self.cpu_device)

        return { # All tensors returned are on CPU
            "initial_states_for_responses": final_initial_states_cpu,
            "responses_actions_tensor": final_actions_tensor_cpu,
            "responses_log_probs_tensor": final_log_probs_tensor_cpu,
            "prompt_indices": final_prompt_indices_cpu
        }

    def compute_group_relative_advantages(self, rewards_tensor_cpu, prompt_indices_tensor_cpu):
        # This computation can stay on CPU
        advantages_cpu = torch.zeros_like(rewards_tensor_cpu, device=self.cpu_device)
        unique_prompt_ids = torch.unique(prompt_indices_tensor_cpu)
        for prompt_id in unique_prompt_ids:
            mask = (prompt_indices_tensor_cpu == prompt_id)
            rewards_for_prompt = rewards_tensor_cpu[mask]
            if rewards_for_prompt.numel() > 0:
                mean_reward = rewards_for_prompt.mean()
                std_reward = rewards_for_prompt.std() + 1e-8
                advantages_cpu[mask] = (rewards_for_prompt - mean_reward) / std_reward
        return advantages_cpu

    def update(self, collected_data_cpu, rewards_for_responses_cpu):
        # actor_old is synced on GPU
        self.actor_old.load_state_dict(self.actor_current.state_dict())
        self.actor_old.eval() # actor_old is on self.device (GPU)

        # Data from collected_data_cpu is on CPU
        initial_states_for_responses_cpu = collected_data_cpu["initial_states_for_responses"]
        old_actions_tensor_cpu = collected_data_cpu["responses_actions_tensor"]
        old_log_probs_tensor_cpu = collected_data_cpu["responses_log_probs_tensor"]
        prompt_indices_cpu = collected_data_cpu["prompt_indices"] # Not directly used in loss, but for advantages

        effective_batch_size = old_actions_tensor_cpu.shape[0]
        seq_len = old_actions_tensor_cpu.shape[1]

        # Advantages computed on CPU, then moved to GPU for loss calculation
        advantages_cpu = self.compute_group_relative_advantages(rewards_for_responses_cpu, prompt_indices_cpu)
        advantages_per_token_cpu = advantages_cpu.unsqueeze(1).repeat(1, seq_len)
        
        # Move tensors needed for loss calculation repeatedly to GPU once
        old_log_probs_tensor_gpu = old_log_probs_tensor_cpu.to(self.device)
        advantages_per_token_gpu = advantages_per_token_cpu.to(self.device)

        # This large tensor holding evolving states for re-evaluation stays on CPU.
        current_eval_states_for_eff_batch_cpu = initial_states_for_responses_cpu.clone().detach()

        for epoch in range(self.k_epochs):
            self.actor_current.train() # actor_current is on self.device (GPU)
            
            all_new_log_probs_time_gpu = []
            all_entropy_time_gpu = []
            all_kl_div_time_gpu = []

            # This copy is for the current epoch's state evolution, starts fresh from initial states (on CPU)
            current_epoch_eval_states_cpu = current_eval_states_for_eff_batch_cpu.clone()

            for t in range(seq_len):
                new_log_probs_at_t_list_chunks_gpu = []
                entropy_at_t_list_chunks_gpu = []
                kl_div_at_t_list_chunks_gpu = []
                
                updated_states_for_current_t_chunks_cpu = [] # Store updated states from chunks on CPU

                for i in range(0, effective_batch_size, self.strict_model_batch_size):
                    start_idx = i; end_idx = min(i + self.strict_model_batch_size, effective_batch_size)
                    
                    # Move only necessary chunks to GPU
                    states_chunk_gpu = current_epoch_eval_states_cpu[start_idx:end_idx].to(self.device)
                    actions_chunk_gpu = old_actions_tensor_cpu[start_idx:end_idx, t].to(self.device)
                    
                    action_dist_current_chunk = self.actor_current.get_action_dist(states_chunk_gpu)
                    new_log_probs_chunk_gpu = action_dist_current_chunk.log_prob(actions_chunk_gpu)
                    new_probs_chunk_gpu = torch.exp(new_log_probs_chunk_gpu)
                    entropy_chunk_gpu = action_dist_current_chunk.entropy()
                    
                    with torch.no_grad(): # actor_reference is on GPU
                        action_dist_ref_chunk = self.actor_reference.get_action_dist(states_chunk_gpu)
                        ref_log_probs_chunk_gpu = action_dist_ref_chunk.log_prob(actions_chunk_gpu)
                        ref_probs_chunk_gpu = torch.exp(ref_log_probs_chunk_gpu)
                    epsilon_div = 1e-8 # 用于防止除以零
                    epsilon_log = 1e-8 # 用于防止 log(0)
                    q_ratio = ref_probs_chunk_gpu / (new_probs_chunk_gpu + epsilon_div)
                    kl_div_chunk_gpu = q_ratio - torch.log(q_ratio + epsilon_log) - 1
                    # kl_div_chunk_gpu = torch.distributions.kl.kl_divergence(action_dist_current_chunk, action_dist_ref_chunk)

                    new_log_probs_at_t_list_chunks_gpu.append(new_log_probs_chunk_gpu)
                    entropy_at_t_list_chunks_gpu.append(entropy_chunk_gpu)
                    kl_div_at_t_list_chunks_gpu.append(kl_div_chunk_gpu)
                    
                    # Update states for this chunk on GPU, then move back to CPU for storage
                    updated_states_chunk_gpu = placeholder_batch_state_update(
                        states_chunk_gpu, actions_chunk_gpu, self.action_dim, self.device
                    )
                    updated_states_for_current_t_chunks_cpu.append(updated_states_chunk_gpu.to(self.cpu_device))

                    del states_chunk_gpu, actions_chunk_gpu, new_log_probs_chunk_gpu, entropy_chunk_gpu, kl_div_chunk_gpu, updated_states_chunk_gpu
                
                # Update the CPU master copy of states for the next time step
                current_epoch_eval_states_cpu = torch.cat(updated_states_for_current_t_chunks_cpu)

                all_new_log_probs_time_gpu.append(torch.cat(new_log_probs_at_t_list_chunks_gpu))
                all_entropy_time_gpu.append(torch.cat(entropy_at_t_list_chunks_gpu))
                all_kl_div_time_gpu.append(torch.cat(kl_div_at_t_list_chunks_gpu))
            
            # These are now large GPU tensors: [eff_batch, seq_len]
            new_log_probs_tensor_gpu = torch.stack(all_new_log_probs_time_gpu, dim=1)
            entropy_tensor_gpu = torch.stack(all_entropy_time_gpu, dim=1)
            kl_div_tensor_gpu = torch.stack(all_kl_div_time_gpu, dim=1)
            
            # Loss calculations on GPU
            ratios_gpu = torch.exp(new_log_probs_tensor_gpu - old_log_probs_tensor_gpu.detach()) # old_log_probs is already on GPU
            surr1_gpu = ratios_gpu * advantages_per_token_gpu.detach() # advantages is already on GPU
            surr2_gpu = torch.clamp(ratios_gpu, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_per_token_gpu.detach()
            
            actor_loss_clipped_gpu = -torch.min(surr1_gpu, surr2_gpu).mean()
            kl_loss_gpu = kl_div_tensor_gpu.mean()
            entropy_bonus_gpu = entropy_tensor_gpu.mean()

            total_loss_gpu = actor_loss_clipped_gpu + self.beta_kl * kl_loss_gpu - self.gamma_entropy * entropy_bonus_gpu

            self.optimizer_actor.zero_grad()
            total_loss_gpu.backward() # Gradients computed on GPU parameters
            self.optimizer_actor.step() # Optimizer updates GPU parameters

            if (epoch % 1 == 0) or (epoch == self.k_epochs -1) :
                print(f"    Epoch [{epoch+1}/{self.k_epochs}], Loss: {total_loss_gpu.item():.4f} (Clip: {actor_loss_clipped_gpu.item():.4f}, KL: {kl_loss_gpu.item():.4f}, Ent: {entropy_bonus_gpu.item():.4f})")
            
            # Clean up large GPU tensors from this epoch's re-evaluation
            del new_log_probs_tensor_gpu, entropy_tensor_gpu, kl_div_tensor_gpu, ratios_gpu, surr1_gpu, surr2_gpu
            del actor_loss_clipped_gpu, kl_loss_gpu, entropy_bonus_gpu, total_loss_gpu
            del all_new_log_probs_time_gpu, all_entropy_time_gpu, all_kl_div_time_gpu # lists of gpu tensors

        del old_log_probs_tensor_gpu, advantages_per_token_gpu # Clean up tensors moved at start of update
        if self.device.type == 'cuda': torch.cuda.empty_cache()


# --- 3. 训练循环 ---
if __name__ == '__main__':
    STATE_DIM = 64; ACTION_DIM = 50; STRICT_MODEL_BATCH_SIZE = 8
    NUM_DISTINCT_PROMPTS_PER_LOGICAL_BATCH = 8
    NUM_SAMPLES_PER_PROMPT = 8
    assert NUM_DISTINCT_PROMPTS_PER_LOGICAL_BATCH <= STRICT_MODEL_BATCH_SIZE
    LR_ACTOR = 3e-5; EPS_CLIP = 0.2; K_EPOCHS = 3; BETA_KL = 0.02; GAMMA_ENTROPY = 0.01
    MAX_RESPONSE_LEN = 10; TRAINING_ITERATIONS = 10 # Shorter run for quick demo

    # Explicitly set to cuda if available, else cpu
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("CUDA is available! Training on GPU.")
    else:
        DEVICE = torch.device("cpu")
        print("CUDA not available. Training on CPU.")

    CPU_DEVICE = torch.device("cpu") # Explicit CPU device for clarity

    print(f"Primary compute device: {DEVICE}")
    print(f"Strict model batch size: {STRICT_MODEL_BATCH_SIZE}")
    # ... (other print statements)

    SFT_MODEL_PATH = "dummy_sft_model_strict_cpu.pth"
    # Save dummy SFT model (parameters will be on CPU by default for PolicyNetwork)
    dummy_sft_policy_cpu = PolicyNetwork(STATE_DIM, ACTION_DIM)
    torch.save(dummy_sft_policy_cpu.state_dict(), SFT_MODEL_PATH)

    grpo_agent = GRPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                           strict_model_batch_size=STRICT_MODEL_BATCH_SIZE,
                           sft_model_path=SFT_MODEL_PATH, lr_actor=LR_ACTOR,
                           eps_clip=EPS_CLIP, k_epochs=K_EPOCHS, beta_kl=BETA_KL,
                           gamma_entropy=GAMMA_ENTROPY, device=DEVICE) # Agent's main device is GPU
    print("GRPO Agent initialized.")

    for iteration in tqdm(range(TRAINING_ITERATIONS), desc="Training Iterations"):
        # Prompts are initially created on CPU
        batch_distinct_initial_states_cpu = placeholder_get_batch_prompts_initial_states(
            NUM_DISTINCT_PROMPTS_PER_LOGICAL_BATCH, STATE_DIM, device_for_initial_tensor=CPU_DEVICE
        )
        
        # actor_old is synced on GPU (its parameters are copied from actor_current on GPU)
        grpo_agent.actor_old.load_state_dict(grpo_agent.actor_current.state_dict())

        # Generation happens, data is collected on CPU
        collected_experience_cpu = grpo_agent.generate_responses_for_batch_prompts(
            batch_distinct_initial_states_cpu, # Pass CPU tensor
            NUM_SAMPLES_PER_PROMPT,
            MAX_RESPONSE_LEN
        )
        
        if collected_experience_cpu["responses_actions_tensor"].numel() == 0: continue
            
        # Rewards computed and remain on CPU
        rewards_cpu = placeholder_batch_reward_function(collected_experience_cpu["responses_actions_tensor"])
        
        # Update happens with selective GPU transfer
        grpo_agent.update(collected_experience_cpu, rewards_cpu) # Pass CPU tensors

    print("\n--- Training Complete ---")