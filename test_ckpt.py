from sed_modeling.encoders.beats_encoder import BEATsEncoder

enc = BEATsEncoder(
    checkpoint="./pretrained/beats/BEATs_full_finetune_best_0_78.pt",
    freeze=False,
)
print("loaded ok, output_dim =", enc.output_dim)

