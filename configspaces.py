import neps


def get_pipeline_space(args):
    if args.config_space == "data_augmentation":
        pipeline_space = dict(
            # crops
            crops_scale_boundary=neps.FloatParameter(
                lower=0.1, upper=0.8, log=False, default=0.4, default_confidence="medium"
            ),
            # local_crops_number=neps.IntegerParameter(
            #     lower=0, upper=8, default=8, default_confidence="medium"
            # ),
            # probabilities
            p_horizontal_crop_1=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.5, default_confidence="medium"
            ),
            p_colorjitter_crop_1=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.8, default_confidence="medium"
            ),
            p_grayscale_crop_1=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.2, default_confidence="medium"
            ),
            p_gaussianblur_crop_1=neps.FloatParameter(
                lower=0., upper=1., log=False, default=1.0, default_confidence="medium"
            ),
            p_solarize_crop_1=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.0, default_confidence="medium"
            ),
            p_horizontal_crop_2=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.5, default_confidence="medium"
            ),
            p_colorjitter_crop_2=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.8, default_confidence="medium"
            ),
            p_grayscale_crop_2=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.2, default_confidence="medium"
            ),
            p_gaussianblur_crop_2=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.1, default_confidence="medium"
            ),
            p_solarize_crop_2=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.2, default_confidence="medium"
            ),
            p_horizontal_crop_3=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.5, default_confidence="medium"
            ),
            p_colorjitter_crop_3=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.8, default_confidence="medium"
            ),
            p_grayscale_crop_3=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.2, default_confidence="medium"
            ),
            p_gaussianblur_crop_3=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.5, default_confidence="medium"
            ),
            p_solarize_crop_3=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.0, default_confidence="medium"
            ),
        )
    elif args.config_space == "training":
        pipeline_space = dict(
            lr=neps.FloatParameter(
                lower=0.00001, upper=0.01, log=True, default=0.0005, default_confidence="medium"
            ),
            out_dim=neps.IntegerParameter(
                lower=1000, upper=100000, log=False, default=65536, default_confidence="medium"
            ),
            momentum_teacher=neps.FloatParameter(
                lower=0.8, upper=1, log=True, default=0.996, default_confidence="medium"
            ),
            warmup_teacher_temp=neps.FloatParameter(
                lower=0.001, upper=0.1, log=True, default=0.04, default_confidence="medium"
            ),
            warmup_teacher_temp_epochs=neps.IntegerParameter(
                lower=0, upper=50, log=False, default=0, default_confidence="medium"
            ),
            weight_decay=neps.FloatParameter(
                lower=0.001, upper=0.5, log=True, default=0.04, default_confidence="medium"
            ),
            weight_decay_end=neps.FloatParameter(
                lower=0.001, upper=0.5, log=True, default=0.4, default_confidence="medium"
            ),
            freeze_last_layer=neps.IntegerParameter(
                lower=0, upper=10, log=False, default=1, default_confidence="medium"
            ),
            warmup_epochs=neps.IntegerParameter(
                lower=0, upper=50, log=False, default=10, default_confidence="medium"
            ),
            min_lr=neps.FloatParameter(
                lower=1e-7, upper=1e-5, log=True, default=1e-6, default_confidence="medium"
            ),
            drop_path_rate=neps.FloatParameter(
                lower=0.01, upper=0.5, log=False, default=0.1, default_confidence="medium"
            ),
            optimizer=neps.CategoricalParameter(
                choices=['adamw', 'sgd', 'lars'], default='adamw', default_confidence="medium"
            ),
            # use_bn_in_head=neps.CategoricalParameter(
            #     choices=[True, False], default=False, default_confidence="medium"
            # ),
            # norm_last_layer=neps.CategoricalParameter(
            #     choices=[True, False], default=True, default_confidence="medium"
            # ),
        )
    elif args.config_space == "groupaugment":
        pipeline_space = dict(
            # Probabilities for crop 1
            p_color_transformations_crop_1=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0.5, default_confidence="medium"
            ),
            p_geometric_transformations_crop_1=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0.5, default_confidence="medium"
            ),
            p_non_rigid_transformations_crop_1=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0, default_confidence="medium"
            ),
            p_quality_transformations_crop_1=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0, default_confidence="medium"
            ),
            p_exotic_transformations_crop_1=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0, default_confidence="medium"
            ),
            #
            # Probabilities for crop 2
            p_color_transformations_crop_2=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0.5, default_confidence="medium"
            ),
            p_geometric_transformations_crop_2=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0.5, default_confidence="medium"
            ),
            p_non_rigid_transformations_crop_2=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0, default_confidence="medium"
            ),
            p_quality_transformations_crop_2=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0, default_confidence="medium"
            ),
            p_exotic_transformations_crop_2=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0, default_confidence="medium"
            ),
            #
            # Probabilities for crop 3
            p_color_transformations_crop_3=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0.5, default_confidence="medium"
            ),
            p_geometric_transformations_crop_3=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0.5, default_confidence="medium"
            ),
            p_non_rigid_transformations_crop_3=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0, default_confidence="medium"
            ),
            p_quality_transformations_crop_3=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0, default_confidence="medium"
            ),
            p_exotic_transformations_crop_3=neps.FloatParameter(
                lower=0, upper=1, log=False, default=0, default_confidence="medium"
            ),
            #
            # Number of transformations per group (same for each crop)
            n_color_transformations=neps.IntegerParameter(
                lower=1, upper=5, log=False, default=1, default_confidence="medium"
            ),
            n_geometric_transformations=neps.IntegerParameter(
                lower=1, upper=2, log=False, default=1, default_confidence="medium"
            ),
            n_non_rigid_transformations=neps.IntegerParameter(
                lower=1, upper=3, log=False, default=1, default_confidence="medium"
            ),
            n_quality_transformations=neps.IntegerParameter(
                lower=1, upper=2, log=False, default=1, default_confidence="medium"
            ),
            n_exotic_transformations=neps.IntegerParameter(
                lower=1, upper=2, log=False, default=1, default_confidence="medium"
            ),
            n_total=neps.IntegerParameter(
                lower=1, upper=5, log=False, default=1, default_confidence="medium"
            ),
        )
    elif args.config_space == "joint":
        pipeline_space = dict(
            # data augmentation
                # crops
            crops_scale_boundary=neps.FloatParameter(
                lower=0.1, upper=0.8, log=False, default=0.4, default_confidence="medium"
            ),
            # local_crops_number=neps.IntegerParameter(
            #     lower=0, upper=8, default=8, default_confidence="medium"
            # ),
                # probabilities
            p_horizontal_crop_1=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.5, default_confidence="medium"
            ),
            p_colorjitter_crop_1=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.8, default_confidence="medium"
            ),
            p_grayscale_crop_1=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.2, default_confidence="medium"
            ),
            p_gaussianblur_crop_1=neps.FloatParameter(
                lower=0., upper=1., log=False, default=1.0, default_confidence="medium"
            ),
            p_solarize_crop_1=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.0, default_confidence="medium"
            ),
            p_horizontal_crop_2=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.5, default_confidence="medium"
            ),
            p_colorjitter_crop_2=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.8, default_confidence="medium"
            ),
            p_grayscale_crop_2=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.2, default_confidence="medium"
            ),
            p_gaussianblur_crop_2=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.1, default_confidence="medium"
            ),
            p_solarize_crop_2=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.2, default_confidence="medium"
            ),
            p_horizontal_crop_3=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.5, default_confidence="medium"
            ),
            p_colorjitter_crop_3=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.8, default_confidence="medium"
            ),
            p_grayscale_crop_3=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.2, default_confidence="medium"
            ),
            p_gaussianblur_crop_3=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.5, default_confidence="medium"
            ),
            p_solarize_crop_3=neps.FloatParameter(
                lower=0., upper=1., log=False, default=0.0, default_confidence="medium"
            ),

            # training
            lr=neps.FloatParameter(
                lower=0.00001, upper=0.01, log=True, default=0.0005, default_confidence="medium"
            ),
            out_dim=neps.IntegerParameter(
                lower=1000, upper=100000, log=False, default=65536, default_confidence="medium"
            ),
            momentum_teacher=neps.FloatParameter(
                lower=0.8, upper=1, log=True, default=0.996, default_confidence="medium"
            ),
            warmup_teacher_temp=neps.FloatParameter(
                lower=0.001, upper=0.1, log=True, default=0.04, default_confidence="medium"
            ),
            warmup_teacher_temp_epochs=neps.IntegerParameter(
                lower=0, upper=50, log=False, default=0, default_confidence="medium"
            ),
            weight_decay=neps.FloatParameter(
                lower=0.001, upper=0.5, log=True, default=0.04, default_confidence="medium"
            ),
            weight_decay_end=neps.FloatParameter(
                lower=0.001, upper=0.5, log=True, default=0.4, default_confidence="medium"
            ),
            freeze_last_layer=neps.IntegerParameter(
                lower=0, upper=10, log=False, default=1, default_confidence="medium"
            ),
            warmup_epochs=neps.IntegerParameter(
                lower=0, upper=50, log=False, default=10, default_confidence="medium"
            ),
            min_lr=neps.FloatParameter(
                lower=1e-7, upper=1e-5, log=True, default=1e-6, default_confidence="medium"
            ),
            drop_path_rate=neps.FloatParameter(
                lower=0.01, upper=0.5, log=False, default=0.1, default_confidence="medium"
            ),
            optimizer=neps.CategoricalParameter(
                choices=['adamw', 'sgd', 'lars'], default='adamw', default_confidence="medium"
            ),
            # use_bn_in_head=neps.CategoricalParameter(
            #     choices=[True, False], default=False, default_confidence="medium"
            # ),
            # norm_last_layer=neps.CategoricalParameter(
            #     choices=[True, False], default=True, default_confidence="medium"
            # ),
        )
    else:
        raise NotImplementedError
    if args.is_multifidelity_run:
        # Add to dict: epoch_fidelity=neps.IntegerParameter(lower=1, upper=args.epochs, is_fidelity=True),
        raise NotImplementedError
    return pipeline_space
