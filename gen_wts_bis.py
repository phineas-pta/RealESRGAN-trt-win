# -*- coding: utf-8 -*-

import argparse
import os
import struct
from RealESRGAN import RealESRGAN

def parseArgs():
	parser = argparse.ArgumentParser(description="convert custom Real-ESRGAN checkpoints from .pth to .wts")
	parser.add_argument("-i", "--input", type=str, help="path containing .pth files")
	parser.add_argument("-o", "--output", type=str, help="path containing .wts files")
	parser.add_argument(
		"-n", "--model_name", type=str, help="Model names",
		choices=["RealESRGAN_x4", "RealESRGAN_x2"]
	)
	parser.add_argument("-v", "--verbose", action="store_true", help="print all checkpoint values")

	return parser.parse_args()

args = parseArgs()

# determine models according to model names
match args.model_name:
	case "RealESRGAN_x4":
		model = RealESRGAN("cuda", scale=4)
	case "RealESRGAN_x2":
		model = RealESRGAN("cuda", scale=2)
	case _:
		raise ValueError("unknown model")

# determine model paths
model_path = os.path.join(args.input, f"{args.model_name}.pth")
if not os.path.isfile(model_path):
	raise ValueError(f"Model {args.model_name} does not exist.")

model.load_weights(model_path, download=False)

wts_file = os.path.join(args.output, f"{args.model_name}.wts")
print("making wts file ...")
f = open(wts_file, "w")
f.write("{}\n".format(len(model.model.state_dict().keys())))
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
