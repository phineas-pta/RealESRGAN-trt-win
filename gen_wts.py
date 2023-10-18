# -*- coding: utf-8 -*-

import argparse
import os
import struct
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def parseArgs():
	parser = argparse.ArgumentParser(description="convert original Real-ESRGAN checkpoints from .pth to .wts")
	parser.add_argument("-i", "--input", type=str, help="path containing .pth files")
	parser.add_argument("-o", "--output", type=str, help="path containing .wts files")
	parser.add_argument(
		"-n", "--model_name", type=str, help="Model names",
		choices=["RealESRGAN_x4plus", "RealESRNet_x4plus", "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x2plus", "realesr-animevideov3"]
	)
	parser.add_argument("--tile", type=int, default=0, help="Tile size, 0 for no tile during testing")
	parser.add_argument("--tile_pad", type=int, default=10, help="Tile padding")
	parser.add_argument("--pre_pad", type=int, default=0, help="Pre padding size at each border")
	parser.add_argument("--fp16", action="store_true", help="Use fp16 precision during inference")
	parser.add_argument("-v", "--verbose", action="store_true", help="print all checkpoint values")

	## disable coz no effect
	# parser.add_argument("-s", "--outscale", type=float, default=4, help="The final upsampling scale of the image")
	# parser.add_argument("--face_enhance", action="store_true", help="Use GFPGAN to enhance face")
	# parser.add_argument("--suffix", type=str, default="out", help="Suffix of the restored image")
	# parser.add_argument("--alpha_upsampler", type=str, default="realesrgan", choices=["realesrgan", "bicubic"], help="The upsampler for the alpha channels")
	# parser.add_argument("--ext", type=str, default="auto", choices=["auto", "jpg", "png"], help="auto means using the same extension as inputs")

	return parser.parse_args()

args = parseArgs()

# determine models according to model names
if args.model_name in ["RealESRGAN_x4plus", "RealESRNet_x4plus"]:  # x4 RRDBNet model
	model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
	netscale = 4
elif args.model_name in ["RealESRGAN_x4plus_anime_6B"]:  # x4 RRDBNet model with 6 blocks
	model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
	netscale = 4
elif args.model_name in ["RealESRGAN_x2plus"]:  # x2 RRDBNet model
	model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
	netscale = 2
elif args.model_name in ["realesr-animevideov3"]:  # x4 VGG-style model (XS size)
	model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
	netscale = 4
else:
	raise ValueError("unknown model")

# determine model paths
model_path = os.path.join(args.input, f"{args.model_name}.pth")
if not os.path.isfile(model_path):
	raise ValueError(f"Model {args.model_name} does not exist.")

# restorer
model = RealESRGANer(
	scale=netscale,
	model_path=model_path,
	model=model,
	tile=args.tile,
	tile_pad=args.tile_pad,
	pre_pad=args.pre_pad,
	half=args.fp16
)

wts_file = os.path.join(args.output, f"{args.model_name}.wts")
print("making wts file ...")
f = open(wts_file, "w")
f.write(f"{len(model.model.state_dict().keys())}\n")
for k, v in model.model.state_dict().items():
	if args.verbose:
		print("key: ", k)
		print("value: ", v.shape)
	vr = v.reshape(-1).cpu().numpy()
	f.write(f"{k} {len(vr)}")
	for vv in vr:
		f.write(" ")
		f.write(struct.pack(">f", float(vv)).hex())
	f.write("\n")
print("Completed wts file!")
