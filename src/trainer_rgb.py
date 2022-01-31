'''
trainer_rgb.py

* Parsing arguments
* Loading/saving checkpoints
* Visualization functions
* Model training & validation
* main function
'''


import argparse


### PARSING ###

def parse_keyword(keyword):
    if isinstance(keyword, bool):
       return keyword
    if keyword.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif keyword.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong keyword.')


parser = argparse.ArgumentParser()
parser.add_argument('--beta',
                        type=float,
                        default=0.25,
                        metavar='DOUBLE',
                        help='Beta hyperparameter'
                    )
parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        metavar='DOUBLE',
                        help='Learning rate hyperparameter'
                    )
parser.add_argument('--experiment',
                        type=int,
                        default=0,
                        metavar='INT',
                        help='Number of experiment'
                    )
parser.add_argument('--summary',
                        type=str,
                        default=None,
                        metavar='STRING',
                        help='Summary to be shown in wandb'
                    )
parser.add_argument('--add_l1_term',
                        type=parse_keyword,
                        default=False,
                        metavar='BOOL',
                        help='Add L1 term in the loss function'
                    )
parser.add_argument('--from_checkpoint',
                        type=parse_keyword,
                        default=False,
                        metavar='BOOL',
                        help='Use checkpoint listed by experiment number'
                    )
parser.add_argument('--transform',
                        type=str,
                        default='cosine',
                        metavar='STR',
                        help='Which transform to use: [cosine] or [fourier]'
                    )
parser.add_argument('--ft_container',
                        type=str,
                        default='mag',
                        metavar='STR',
                        help='If [fourier], container to use: [mag], [phase], [magphase]'
                    )
parser.add_argument('--mp_encoder',
                        type=str,
                        default='single',
                        metavar='STR',
                        help='If [fourier] and [magphase], type of magphase encoder: [single], [double]'
                    )
parser.add_argument('--mp_decoder',
                        type=str,
                        default='double',
                        metavar='STR',
                        help='If [fourier] and [magphase], type of magphase encoder: [unet], [double]'
                    )
parser.add_argument('--mp_join',
                        type=str,
                        default='mean',
                        metavar='STR',
                        help='If [fourier] and [magphase] and [decoder=double], type of join operation: [mean], [2D], [3D]'
                    )


if __name__ == '__main__':

    args = parser.parse_args()
    print(args)

'''
    train_loader = loader(
        set='train',
        rgb=args.rgb,
        transform=args.transform,
    )
    test_loader = loader(
        set='test',
        rgb=args.rgb,
        transform=args.transform,
    )

    model = StegoUNet(
        # architecture=args.architecture,
        transform=args.transform,
        add_noise=args.add_noise,
        noise_kind=args.noise_kind,
        noise_amplitude=args.noise_amplitude,
        phase_type=args.phase_type
    )

    if args.from_checkpoint:
        # Load checkpoint
        checkpoint = torch.load(os.path.join(os.environ.get('OUT_PATH'),f'models/             checkpoint_run_{args.experiment}.pt'), map_location='cpu')
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
        print('Checkpoint loaded ++')
    train(
        model=model,
        tr_loader=train_loader,
        vd_loader=test_loader,
        beta=args.beta,
        lr=args.lr,
        epochs=2,
        slide=15,
        prev_epoch=checkpoint['epoch'] if args.from_checkpoint else None,
        prev_i=checkpoint['i'] if args.from_checkpoint else None,
        summary=args.summary,
        experiment=args.experiment,
        add_l1_term=args.add_l1_term,
        transform=args.transform,
        on_phase=args.on_phase,
        phase_type=args.phase_type
    )
'''
