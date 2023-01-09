from configs.base.loveda import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='RSSFormer',

        params=dict(
            backbone=dict(
                hrnet_type='hrnetv2_w32',
                pretrained=True,
                norm_eval=False,
                frozen_stages=-1,
                with_cp=False,
                with_gc=False,
            ),
            neck=dict(
                in_channels=480,
            ),
            classes=7,
            head=dict(
                in_channels=480,
                upsample_scale=4.0,
            ),
            loss=dict(
                ignore_index=-1,
                ce=dict(),
                #diceloss=dict(scaler=0.5),
                #bceloss=dict(scaler=0.7),
                #tverloss=dict(scaler=0.5,alpha=0.5,beta=0.5),
                #fcloss=dict(gamma=5.0),

            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)
