import os
import argparse
import torch
import torch.nn as nn
import torchreid
from torchreid.tools.extract_part_based_features import extract_reid_features
from torchreid.data.masks_transforms import compute_parts_num_and_names
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity, Writer, load_checkpoint
)

from torchreid.scripts.default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs, display_config_diff
)
from torchreid.utils.engine_state import EngineState


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, stage, datamanager, model, optimizer, scheduler, writer, engine_state):
    if stage == 'stage1':
        engine = torchreid.engine.ImagePartBasedEngine1(
            datamanager,
            model,
            optimizer=optimizer,
            loss_name=cfg.loss.part_based.name,
            config=cfg,
            margin=cfg.loss.triplet.margin,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            save_model_flag=cfg.model.save_model_flag,
            writer=writer,
            engine_state=engine_state,
            dist_combine_strat=cfg.test.part_based.dist_combine_strat,
            batch_size_pairwise_dist_matrix=cfg.test.batch_size_pairwise_dist_matrix,
            mask_filtering_training=cfg.model.bpbreid.mask_filtering_training,
            mask_filtering_testing=cfg.model.bpbreid.mask_filtering_testing
        )
    elif stage == 'stage2':
        engine = torchreid.engine.ImagePartBasedEngine2(
            datamanager,
            model,
            optimizer=optimizer,
            loss_name=cfg.loss.part_based.name,
            config=cfg,
            margin=cfg.loss.triplet.margin,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            save_model_flag=cfg.model.save_model_flag,
            writer=writer,
            engine_state=engine_state,
            dist_combine_strat=cfg.test.part_based.dist_combine_strat,
            batch_size_pairwise_dist_matrix=cfg.test.batch_size_pairwise_dist_matrix,
            mask_filtering_training=cfg.model.bpbreid.mask_filtering_training,
            mask_filtering_testing=cfg.model.bpbreid.mask_filtering_testing
        )
    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.save_dir:
        cfg.data.save_dir = args.save_dir
    if args.inference_enabled:
        cfg.inference.enabled = args.inference_enabled
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms
    if args.job_id:
        cfg.project.job_id = args.job_id


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        '--save_dir', type=str, default='', help='path to output root dir'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    parser.add_argument(
        '--job-id',
        type=int,
        default=None,
        help='Slurm job id'
    )
    parser.add_argument(
        '--inference-enabled',
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    cfg1, cfg2 = build_config(args, args.config_file)

    engine1, model, datamanger = build_torchreid_model_engine_stage1(cfg1)
    print('Starting experiment {} with job id {} and creation date {}'.format(cfg1.project.experiment_id,
                                                                              cfg1.project.job_id,
                                                                              cfg1.project.start_time))
    engine1.run(**engine_run_kwargs(cfg1))
    print('Starting stage2 traning process.')
    engine2, model = build_torchreid_model_engine_stage2(cfg2, datamanger, model)
    engine2.run(**engine_run_kwargs(cfg2))
    print(
        'End of experiment {} with job id {} and creation date {}'.format(cfg1.project.experiment_id, cfg1.project.job_id,
                                                                          cfg1.project.start_time))

    if cfg1.inference.enabled:
        print("Starting inference on external data")
        extract_reid_features(cfg1, cfg1.inference.input_folder, cfg1.data.save_dir, model)




def build_config(args=None, config_file=None, config=None):
    cfg = get_default_config()
    default_cfg_copy = cfg.clone()
    cfg.use_gpu = torch.cuda.is_available()
    if config:
        cfg.merge_from_other_cfg(config)
    if config_file:
        cfg.merge_from_file(config_file)
        cfg.project.config_file = os.path.basename(config_file)
    if args is not None:
        reset_config(cfg, args)
        cfg.merge_from_list(args.opts)
    # set parts information (number of parts K and each part name),
    # depending on the original loaded masks size or the transformation applied:
    compute_parts_num_and_names(cfg)
    if cfg.model.load_weights and check_isfile(cfg.model.load_weights) and cfg.model.load_config:
        checkpoint = load_checkpoint(cfg.model.load_weights)
        # if 'config' in checkpoint and False:
        if 'config' in checkpoint:
            print('Overwriting current config with config loaded from {}'.format(cfg.model.load_weights))
            bpbreid_config = checkpoint['config'].model.bpbreid
            if checkpoint['config'].data.sources[0] != cfg.data.targets[0]:
                print('WARNING: the train dataset of the loaded model is different from the target dataset in the '
                      'current config.')
            bpbreid_config.pop('hrnet_pretrained_path', None)
            bpbreid_config.masks.pop('dir', None)
            cfg.model.bpbreid.merge_from_other_cfg(bpbreid_config)
        else:
            print('Could not load config from file {}'.format(cfg.model.load_weights))
    display_config_diff(cfg, default_cfg_copy)
    cfg.data.save_dir = os.path.join(cfg.data.save_dir, str(cfg.project.job_id))
    os.makedirs(cfg.data.save_dir)
    cfg_stage2 = cfg.clone()
    cfg_stage1 = merge_stage1_config(cfg)
    return cfg_stage1, cfg_stage2

def merge_stage1_config(cfg):
    cfg.train.optim= cfg.train.stage1.optim
    cfg.train.lr = cfg.train.stage1.lr
    cfg.train.weight_decay = cfg.train.stage1.weight_decay
    cfg.train.lr_scheduler = cfg.train.stage1.lr_scheduler
    cfg.train.stepsize = cfg.train.stage1.stepsize
    cfg.train.gamma = cfg.train.stage1.gamma
    cfg.train.max_epoch = cfg.train.stage1.max_epoch
    cfg.train.start_epoch = cfg.train.stage1.start_epoch
    return cfg

def build_torchreid_model_engine_stage1(cfg):
    if cfg.project.debug_mode:
        torch.autograd.set_detect_anomaly(True)
    logger = Logger(cfg)
    writer = Writer(cfg)
    set_random_seed(cfg.train.seed)
    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True
    datamanager = build_datamanager(cfg)
    engine_state = EngineState(cfg.train.start_epoch, cfg.train.max_epoch)
    writer.init_engine_state(engine_state, cfg.model.bpbreid.masks.parts_num)
    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
        config=cfg
    )
    logger.add_model(model)

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()
    optimizer = torchreid.optim.build_optimizer_stage1(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )
    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )
    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, 'stage1',datamanager, model, optimizer, scheduler, writer, engine_state)
    return engine, model, datamanager

def build_torchreid_model_engine_stage2(cfg, datamanager, model):
    if cfg.project.debug_mode:
        torch.autograd.set_detect_anomaly(True)
    logger = Logger(cfg)
    writer = Writer(cfg)
    set_random_seed(cfg.train.seed)
    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True
    engine_state = EngineState(cfg.train.start_epoch, cfg.train.max_epoch)
    writer.init_engine_state(engine_state, cfg.model.bpbreid.masks.parts_num)
    print('Building model: {}'.format(cfg.model.name))
    logger.add_model(model)
    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )
    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg,'stage2', datamanager, model, optimizer, scheduler, writer, engine_state)
    return engine, model


if __name__ == '__main__':
    main()
