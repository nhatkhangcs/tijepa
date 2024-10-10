from src.masks.multiblock import MaskCollator

masker = MaskCollator(
    input_size=(224, 224),
    patch_size=16,
    enc_mask_scale=(0.2, 0.8),
    pred_mask_scale=(0.2, 0.8),
    aspect_ratio=(0.3, 3.0),
    nenc=1,
    npred=4,
    min_keep=4,
    allow_overlap=False
)

# collated_batch, collated_masks_enc, collated_masks_pred = masker(batch=[1])
# print(collated_batch)
# print(collated_masks_enc)
# print(collated_masks_pred)
# print(f'{collated_batch.shape=}')
# print(f'{len(collated_masks_enc)=}')
# print(f'{len(collated_masks_pred)=}')

# print()
# collated_batch, collated_masks_enc, collated_masks_pred = masker(batch=[1, 1])
# print(collated_batch)
# print(collated_masks_enc)
# print(collated_masks_pred)
# print(f'{collated_batch.shape=}')
# print(f'{len(collated_masks_enc)=}')
# print(f'{len(collated_masks_pred)=}')

print()
collated_batch, collated_masks_enc, collated_masks_pred = masker(batch=[1, 2, 1])
print(collated_batch)
print(collated_masks_enc)
print(collated_masks_pred)
print(f'{collated_batch.shape=}')
print(f'{len(collated_masks_enc)=}')
print(f'{len(collated_masks_pred)=}')
print(f'{len(collated_masks_enc[0])=}')
print(f'{len(collated_masks_pred[0])=}')

print("\n\n")

i = 1
ENC = set(collated_masks_enc[0].tolist()[i])
PREDS = []
for j in range(4):
    PREDS.append(collated_masks_pred[j].tolist()[i])

[print(len(p)) for p in PREDS]
PRED_SET = set()
[PRED_SET.update(p) for p in PREDS]
print(ENC)
print(f"{len(ENC)=}")
print(PRED_SET)
print(f"{len(PRED_SET)=}")

PRED_SET.update(ENC)
print(len(PRED_SET))

ALL = set([i for i in range(16*16)])
print(ALL - PRED_SET - ENC)
