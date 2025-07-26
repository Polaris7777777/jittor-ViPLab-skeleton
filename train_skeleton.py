import sched
import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn
from jittor import optim, lr_scheduler

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix, SamplerFPS, SamplerFPSMix, FPSWrapper
from dataset.exporter import Exporter
from models.skeleton import create_model

from models.metrics import J2J, bone_length_symmetry_loss, joint_symmetry_loss

# import tensorboardX as SummaryWriter

# Set Jittor flags
jt.flags.use_cuda = 1


def train(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    # writer = SummaryWriter(logdir=args.output_dir)
    
    def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    # Log training parameters
    log_message(f"Starting training with parameters: {args}")
    
    # Create model
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type
    )
    
    # Create sampler based on args
    if args.sampler_type == 'mix':
        sampler = SamplerMix(num_samples=args.num_samples, vertex_samples=args.vertex_samples)
    elif args.sampler_type == 'fps':
        # sampler = SamplerFPS(num_samples=args.num_samples)
        sampler = FPSWrapper(num_samples=args.num_samples)
    elif args.sampler_type == 'fpsmix':
        sampler = SamplerFPSMix(num_samples=args.num_samples, fps_samples=args.vertex_samples)
    else:
        raise ValueError(f"Unknown sampler type: {args.sampler_type}")
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # Create optimizer
    # 将参数分组，对不同类型的参数应用不同的权重衰减策略
    decay_params = []
    no_decay_params = []
    
    # 遍历所有参数，将参数分组
    for name, param in model.named_parameters():
        # 根据参数选项决定是否对偏置和BN参数应用权重衰减
        if 'bn' in name or 'norm' in name:
            no_decay_params.append(param)
        elif 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # 为不同组的参数设置不同的优化选项
    param_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(param_groups, lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = optim.Adam(param_groups, lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # create lr scheduler
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    elif args.scheduler == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90, 120, 150, 180, 210, 240, 270, 300], gamma=args.lr_decay)

    # Create loss function
    criterion = nn.MSELoss()
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        transform=transform,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            transform=transform,
        )
    else:
        val_loader = None
    λ1 = args.bone_λ
    λ2 = args.sym_λ
# Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints = data['vertices'], data['joints'] # [B N 3]
            vertices = vertices.permute(0, 2, 1)  # [B, 3, N]
            outputs = model(vertices)
            joints = joints.reshape(outputs.shape[0], -1)
            loss_train = criterion(outputs, joints)
            
            bone_loss_train = bone_length_symmetry_loss(outputs.reshape(outputs.shape[0], -1, 3))
            sym_loss_train = joint_symmetry_loss(outputs.reshape(outputs.shape[0], -1, 3))
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss_train + λ1 * bone_loss_train + λ2 * sym_loss_train)
            optimizer.step()
            
            # Calculate statistics
            train_loss += loss_train.item() + λ1 * bone_loss_train.item() + λ2 * sym_loss_train.item()
            
            # Print progress
            # if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
            #     log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
            #                f"Loss: {loss.item():.4f}")
        
        scheduler.step()  # Step the scheduler

        # Calculate epoch statistics
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        
        # log_message(f"Epoch [{epoch+1}/{args.epochs}] "
        #            f"Train Loss: {train_loss:.4f} "
        #            f"Time: {epoch_time:.2f}s "
        #            f"LR: {optimizer.lr:.6f}")
        
        # writer.add_scalar('train/loss', train_loss, epoch)
        # writer.add_scalar('train/lr', optimizer.lr, epoch)
        
        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss = 0.0
            J2J_loss = 0.0
            
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels
                vertices, joints = data['vertices'], data['joints']
                joints = joints.reshape(joints.shape[0], -1)
                
                # Reshape input if needed
                if vertices.ndim == 3:  # [B, N, 3]
                    vertices = vertices.permute(0, 2, 1)  # [B, 3, N]
                
                # Forward pass
                outputs = model(vertices)
                loss_val = criterion(outputs, joints)
                # Calculate bone length symmetry loss
                bone_loss_val = bone_length_symmetry_loss(outputs.reshape(outputs.shape[0], -1, 3))
                sym_loss_val = joint_symmetry_loss(outputs.reshape(outputs.shape[0], -1, 3))
                # export render results
                if batch_idx == show_id:
                    exporter = Exporter()
                    # export every joint's corresponding skinning
                    from dataset.format import parents
                    exporter._render_skeleton(path=f"{args.output_dir}/tmp/skeleton/epoch_{epoch}/skeleton_ref.png", joints=joints[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_skeleton(path=f"{args.output_dir}/tmp/skeleton/epoch_{epoch}/skeleton_pred.png", joints=outputs[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_pc(path=f"{args.output_dir}/tmp/skeleton/epoch_{epoch}/vertices.png", vertices=vertices[0].permute(1, 0).numpy())

                val_loss += loss_val.item() + λ1 * bone_loss_val.item() + λ2 * sym_loss_val.item()
                for i in range(outputs.shape[0]):
                    J2J_loss += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item() / outputs.shape[0]
            
            # Calculate validation statistics
            val_loss /= len(val_loader)
            J2J_loss /= len(val_loader)
            # scheduler.step(J2J_loss)  # Step the scheduler based on validation loss
            log_message(f"Epoch [{epoch+1}/{args.epochs}] Time: {epoch_time:.2f}s Train Loss: {loss_train.item():.4f}  Sym Loss: {sym_loss_train.item():.4f} Bone Loss: {bone_loss_train.item():.4f}  LR: {optimizer.lr:.6f}  Val Loss: {loss_val.item():.4f} Sym Loss: {sym_loss_val.item():.4f} Bone Loss: {bone_loss_val.item():.4f}  J2J_Loss: {J2J_loss:.4f}")
            # writer.add_scalar('val/loss', val_loss, epoch)
            # writer.add_scalar('val/J2J_loss', J2J_loss, epoch)

            # Save best model
            if J2J_loss < best_loss:
                best_loss = J2J_loss
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}, best J2J loss: {best_loss:.6f}")
    
    return model, best_loss

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, required=True,
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adamw'],
                        help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'multistep'],
                        help='Learning rate scheduler type')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--lr_step', type=int, default=20,
                        help='Step size for learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.8,
                        help='Decay factor for learning rate')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/skeleton',
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    
    # Sampling parameters
    parser.add_argument('--sampler_type', type=str, default='mix',
                        choices=['mix', 'fps', 'fpsmix'],
                        help='Type of sampler to use')
    parser.add_argument('--num_samples', type=int, default=2048,
                        help='Number of samples for sampling')
    parser.add_argument('--vertex_samples', type=int, default=1024,
                        help='Number of vertex samples for mix sampler')
    
    parser.add_argument('--bone_λ', type=float, default=0.0,
                        help='Weight for bone symmetry loss')
    parser.add_argument('--sym_λ', type=float, default=0.0,
                        help='Weight for joint symmetry loss')
    
    args = parser.parse_args()
    
    # Start training
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()