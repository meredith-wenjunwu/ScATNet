from config.opts import get_config
from bin_data.binarize import main_binarize
from bin_data.binarize_vis import generate_crop, generate_grad, generate_attn


if __name__ == "__main__":
    opts, parser = get_config()
    if opts.generate_crop:
        generate_crop(opts=vars(opts))
    elif opts.save_top_k > 0:
        generate_grad(opts=vars(opts))
    elif opts.save_attn:
        opts.overlay_img_dir = '/projects/patho2/melanoma_diagnosis/x2.5/segmented'
        generate_attn(opts=vars(opts))
    else:
        main_binarize(opts=vars(opts))

