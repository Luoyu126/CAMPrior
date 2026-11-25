import argparse
import os
import random
import torch
import numpy as np
from pathlib import Path
from solvers import *
from solvers.cam_prior import CAMPrior, add_cam_prior_args  # local addition (see UPSTREAM.md)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Diffusion CO Solvers")

    # General settings
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--train_samples", type=int, default=-1)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--valid_samples", type=int, default=-1)   
    parser.add_argument("--test_path", type=str, default='./')
    parser.add_argument("--test_samples", type=int, default=-1)  
    parser.add_argument("--problem", type=str, default='mis', choices=['mis', 'mcl', 'mcut', 'mds', 'tsp', 'cvrp'])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default='./')
    parser.add_argument("--log_file", type=str, default='log.txt')
    parser.add_argument('--wandb_name', type=str, default='CO')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    
    # General RLD sampling hyperparameters
    parser.add_argument("--num_t", type=int, default=1000, help='number of inference sampling steps')
    parser.add_argument("--num_k", type=int, default=100, help='inference sampling group size')
    parser.add_argument("--repetition", type=int, default=1, help='repeated inferene process')
    parser.add_argument("--num_d", type=int, default=30, help='step size')
    parser.add_argument("--tau0", type=float, default=0.1, help='initial temperature')
    parser.add_argument("--tau1", type=float, default=0.0, help='final temperature')
    parser.add_argument('--temp_schedule', default='linear', choices = ["linear", "cosine", "exp"], help='Define the Annealing Schedule')
    parser.add_argument("--lambd", type=float, default=0.0, help='regularization coefficient')
    parser.add_argument("--beta", type=float, default=1.02, help='penalty coefficient')
    parser.add_argument('--inference_schedule', type=str, default='linear', choices=['cosine', 'linear'])
    # local addition (see UPSTREAM.md): DIFUSCO inference starts at noise level t_start (default num_tp)
    parser.add_argument('--inference_t_start', type=int, default=None)
    parser.add_argument('--skip_decode', action='store_true', help='whether skip decoding')
    
    # TSP args
    parser.add_argument("--sparse_factor", type=int, default=50, help='sparse_factor')
    parser.add_argument("--two_opt_iterations", type=int, default=100, help='two_opt_iterations')

    # MIS args
    parser.add_argument("--mis_decoding", type=str, default='continuous', choices=['continuous', 'discrete'])
    
    # Neural model args
    parser.add_argument("--method", type=str, default='RLNN')
    parser.add_argument("--model", type=str, default='AnGNN', help='model architecture')
    parser.add_argument("--num_tp", type=int, default=50, help='number of training sampling steps')
    parser.add_argument("--num_kp", type=int, default=20, help='training sampling group size')
    parser.add_argument("--num_l", type=int, default=1000, help='number of layers')
    parser.add_argument("--num_h", type=int, default=100, help='hidden dimension')
    parser.add_argument("--time_scale", type=int, default=1000, help='time scale')
    parser.add_argument('--diffusion_schedule', type=str, default='linear', choices=['cosine', 'linear'])
    # local addition (see UPSTREAM.md item 42): energy-heated DIFUSCO forward and re-noising sampler
    parser.add_argument('--heat_lambda', type=float, default=0.0, help='flip prob beta_t*sigmoid(lambda_t*dE); 0 = uniform')
    parser.add_argument('--heat_until', type=float, default=0.5, help='lambda_t decays linearly to 0 at heat_until*num_tp')
    parser.add_argument('--difusco_sampler', type=str, default='posterior', choices=['posterior', 'renoise', 'renoise_decode', 'renoise_feasible'])
    parser.add_argument('--ul_chain_decode', action='store_true', help='TSP, UL flip sampler: merge + --decode_two_opt 2-opt after every step (UPSTREAM.md item 63)')
    parser.add_argument('--decode_two_opt', type=int, default=0, help='TSP renoise_decode: 2-opt iterations on each decoded x0 (UPSTREAM.md item 62)')
    # local addition (UPSTREAM.md item 44): several optimal labels per training graph (DIFUSCO)
    parser.add_argument('--label_set', type=str, default=None, help='scripts/mis/build_solution_sets.py output')
    parser.add_argument('--label_choice', type=str, default='first', choices=['first', 'random', 'nearest', 'tilted', 'soft'])
    # local addition (UPSTREAM.md item 69): Schrödinger-bridge coupling (docs/20261125/math_route1.md)
    parser.add_argument('--sb_coupling', action='store_true', help='DIFUSCO: SB coupling between Bernoulli(0.5)^n and the label set')
    parser.add_argument('--sb_omega', type=float, default=0.0, help='SB cost strength omega = log((1+a_T)/(1-a_T)); 0 = independent')
    parser.add_argument('--sb_logg', type=str, default=None, help='scripts/sb/precompute_logg.py output for --sb_omega (training)')
    parser.add_argument('--label_temp', type=float, default=1.0, help="temperature of --label_choice tilted (in set-size units)")
    # local addition (UPSTREAM.md item 45): per-graph weights of the DIFUSCO loss
    parser.add_argument('--graph_weight', type=str, default='none', choices=['none', 'nsets', 'ce'])
    parser.add_argument('--graph_weight_sets', type=str, default=None, help='label set file giving the optimal-set counts')
    parser.add_argument('--graph_weight_power', type=float, default=0.5, help="nsets: w = (number of optima)^-power")
    # local addition (UPSTREAM.md item 43): FM from an MCMC start bank (scripts/mis/build_start_bank.py)
    parser.add_argument('--fm_bank_train', type=str, default=None, help='start bank of the training graphs')
    parser.add_argument('--fm_bank_eval', type=str, default=None, help='start bank of the valid (training) or test graphs')
    parser.add_argument('--fm_bridge_lambda', type=float, default=0.0, help='flip rate of the uniform reference; 0 = mixture path')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr_scheduler', type=str, default='cosine-decay')
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument("--data_precision", type=str, default='fp32', choices=['fp32', 'bf16', 'fp16']) # only matters for RLSA
    parser.add_argument('--do_train', action='store_true', help='whether do training')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint (.pt) to load; used for inference or to resume training')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="local addition (see UPSTREAM.md): continue a run from this epoch; with "
                             "--checkpoint epoch_<start_epoch-1>.pt the temperature and learning-rate "
                             "schedules resume where the run left them (optimizer moments start fresh)")
    parser.add_argument("--stop_epoch", type=int, default=None,
                        help="local addition (see UPSTREAM.md): stop after this many epochs while the "
                             "schedules still run over --epochs (for continuations)")
    parser.add_argument("--save_every_epoch", action="store_true",
                        help="local addition (see UPSTREAM.md): also save epoch_XXXX.pt every epoch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument('--use_activation_checkpoint', action='store_true', help='use activation checkpoint')
    parser.add_argument('--compile', action='store_true', help='torch.compile the model')

    # CAM setting
    parser.add_argument("--samples_per_epoch", type=int, default=1000, help='samples per epoch')
    parser.add_argument("--num_clips", type=int, default=1)

    # DiffUCO setting
    parser.add_argument("--N_warmup", type=int, default=0, help='warmup steps')
    parser.add_argument("--N_anneal", type=int, default=1000, help='annealing steps')
    parser.add_argument("--N_equil", type=int, default=0, help='equilibrium steps')
    parser.add_argument("--noise_energy", type=str, default='bernoulli', choices=['bernoulli', 'annealing'])
    
    # SDDS setting
    parser.add_argument("--inner_loop_steps", type=int, default=1, help='inner loop steps')
    parser.add_argument("--batches_per_step", type=int, default=None, help='max batches from batch_loader per inner loop step; None means all batches')

    add_cam_prior_args(parser)  # local addition (see UPSTREAM.md): --prior_* flags of CAMPrior
    args = parser.parse_args()
    if args.do_train and int(os.environ.get("LOCAL_RANK", 0)) == 0 and os.path.isdir(args.save_dir):
        ckpts = list(Path(args.save_dir).glob('*.pt'))
        if ckpts:
            print(f"Warning: checkpoint(s) found in {args.save_dir}: {[c.name for c in ckpts]}")
            # Local patch (see UPSTREAM.md): fail fast instead of an interactive input() prompt,
            # so unattended runs never hang; set CAM_ALLOW_OVERWRITE=1 to overwrite on purpose.
            if os.environ.get("CAM_ALLOW_OVERWRITE") != "1":
                raise SystemExit(f"Refusing to overwrite checkpoint(s) in {args.save_dir}. "
                                 "Use a new --save_dir, remove --do_train for testing, "
                                 "or set CAM_ALLOW_OVERWRITE=1.")
            print("CAM_ALLOW_OVERWRITE=1: existing checkpoints will be overwritten.")
    if not args.do_train:
        args.wandb_run_name = 'test_' + args.wandb_run_name 
    # fix seeds
    set_seed(args.seed)
    if args.method == 'RLNN':
        solver = RLNN(args)
    elif args.method == 'CAM':
        solver = CAM(args)
    elif args.method == 'CAMPrior':
        solver = CAMPrior(args)
    elif args.method == 'RLSA':
        solver = RLSA(args)
    elif args.method == 'FM':
        solver = FlowMatching(args)
    elif args.method == 'DiffUCO':
        solver = DiffUCO(args)
    elif args.method == 'DIFUSCO':
        solver = DIFUSCO(args)
    elif args.method == 'SDDS':
        solver = SDDS(args)
    else:
        raise NotImplementedError
        
    solver.prepare()

    if args.do_train:
        os.makedirs(args.save_dir, exist_ok = True)
        solver.train(args.save_dir)
    else:
        os.makedirs(Path(args.log_file).parent, exist_ok = True)
        solver.evaluate('test', log_path=args.log_file)
    
if __name__=='__main__':
    main()