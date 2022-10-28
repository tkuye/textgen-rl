import argparse
from env import TextGym
from transformers import  BertTokenizer
from ppo import PPO
from sac import SoftActorCritic

def reward_fn(state, info):
    if "reward_count" not in info:
        info["reward_count"] = 0
    else:
        info["reward_count"] += 1
    
    if "hello" in state:
        return 1
    else:
        return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--n_actions', type=int, default=50257)
    parser.add_argument('--input_shape', type=int, default=768)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--train-type', type=str, default='ppo')
    parser.add_argument('--policy-file-path', type=str, default='policy.pt')
    parser.add_argument('--qf-file-path', type=str, default='critic.pt')
    parser.add_argument('--train', type=bool, default=True)

    parser.add_argument('--buffer-size', type=int, default=100000)
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    env = TextGym(size=args.n_actions, max_length=args.max_length, tokenizer=tokenizer)
    if args.train_type == 'ppo':
        PPO(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.n_epochs,
            env=env,
            gamma=args.gamma,
            value_epochs=args.n_epochs,
            policy_epochs=args.n_epochs,
            max_steps=args.n_episodes,
            reward_fn=reward_fn,
        )
       
    elif args.train_type == 'sac':
        SoftActorCritic(
            epochs=args.n_epochs,
            batch_size=args.batch_size,
            env=env,
            gamma=args.gamma,
            policy_lr=args.lr,
            qf_lr=args.lr,
            policy_file_path=args.policy_file_path,
            qf_file_path=args.qf_file_path,
            buffer_size=args.buffer_size,
            reward_fn=reward_fn,
        )
    
    else:
        print("Invalid training type")

if __name__ == '__main__':
    main()
        

