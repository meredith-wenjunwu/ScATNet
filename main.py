from config.build import *
from experiment.experiment import *
import json
from config.opts import get_config
from utilities.util import save_arguments
from os.path import basename, dirname, splitext
import pdb

def main(args):
    # configuration for experiment


    # -----------------------------------------------------------------------------
    # Preparing Dataset
    # -----------------------------------------------------------------------------
    seed_everything(args)
    train_loader, valid_loader, test_loader = build_dataset(args)
    args = build_class_weights(args)


    # -----------------------------------------------------------------------------
    # Model
    #   - setup model
    #   - load state dict if resume is chosen
    #   - gpu setup and data parallel
    # -----------------------------------------------------------------------------
    args = build_cuda(args)
    criterion = build_criteria(args)
    model, feature_extractor = build_model(args)


    # -----------------------------------------------------------------------------
    # Experiment Setup
    #   - setup visdom and logger
    #   - calculate class weights, setup loss function
    #   - setup optimizer, scheduler
    # -----------------------------------------------------------------------------

    args = build_visualization(args)
    engine = experiment_engine(train_loader, valid_loader,
                               test_loader, **args)
    if opts.mode != 'kmeans':
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)

    # -----------------------------------------------------------------------------
    # Training and Evaluation
    # -----------------------------------------------------------------------------

    if args['mode'] == 'train':
        print_info_message('Training Process Starts...')
        print_info_message("Number of Parameters: {:.2f} M".format(sum([p.numel() for p in model.parameters()])/1e6))
        engine.train(model, args['epochs'], criterion,
                     optimizer, scheduler,
                     args['start_epoch'], feature_extractor=feature_extractor)
    elif args['mode'] == 'test':
        print_info_message('Evaluation on Test Process Starts...')
        engine.eval(model, criterion, mode='test',
                    feature_extractor=feature_extractor)
    elif args['mode'] == 'valid':
        print_info_message('Evaluation on Validation Process Starts...')
        engine.eval(model, criterion, mode='val', feature_extractor=feature_extractor)
    elif args['mode'] == 'valid-train':
        print_info_message('Evaluation on Training Process Starts...')
        engine.eval(model, criterion, mode= 'train', feature_extractor=feature_extractor)


if __name__ == '__main__':
    opts, parser = get_config()
    if opts.resize1 is None:
        resize1 = ['real', 'real']
    else:
        resize1 = opts.resize1
    argument_fname = '{}_cropsize_{}x{}_class_{}_{}_{}'.format('config', resize1[0], resize1[1],
                                                               basename(dirname(opts.data)),
                                                               opts.model, opts.mode)
    if opts.resume is not None:
        model_name = splitext(basename(opts.resume))[0]
    else:
        model_name = 'scratch'

    if opts.binarize:
        attn = 'self_attention' if opts.self_attention else 'weighted'
        opts.save_name = '{}scale_{}_{}x{}_{}_dropout{}'.format(len(opts.resize1_scale),
                                                                basename(dirname(opts.data)),
                                                                opts.model_dim,
                                                                opts.n_layers,
                                                                attn,
                                                                opts.drop_out)

    else:
        opts.save_name = '{}_{}_{}x{}_{}_{}scale_transform{}'.format(opts.base_extractor,
                                                                     basename(dirname(opts.data)),
                                                                     resize1[0], resize1[1],
                                                                     opts.resize2,
                                                                     len(opts.resize1_scale),
                                                                     opts.transform)
    save_arguments(args=opts, save_loc=opts.model_dir, json_file_name=argument_fname)
    print_log_message('Arguments')
    print(json.dumps(vars(opts), indent=4, sort_keys=True))
    main(vars(opts))

