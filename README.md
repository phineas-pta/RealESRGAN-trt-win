# Test: Real-ESRGAN with TensorRT

originals code which only works on linux:
- https://github.com/yester31/Real_ESRGAN_TRT
- https://github.com/wang-xinyu/tensorrtx/tree/master/real-esrgan
- https://github.com/wang-xinyu/tensorrtx/issues/1085#issuecomment-1229422132
- https://github.com/wang-xinyu/tensorrtx/blob/master/tutorials/run_on_windows.md
- https://github.com/yester31/TensorRT_API/tree/master/TensorRT (windows version by author but lost option to select model)

my fork is an attempt to run natively on windows

only a proof-of-concept, no plan to maintain this

same procedure can be applied to https://github.com/bychen7/Face-Restoration-TensorRT (only need adapt file `CMakeLists.txt`)

## License

![LICENSE](https://www.gnu.org/graphics/gplv3-with-text-136x68.png)

3 header files:
- `dirent.h` from https://github.com/tronkko/dirent/blob/master/include/dirent.h
- `getopt.h` from https://gist.github.com/ashelly/7776712#file-getopt-h
- `unistd.h` from https://stackoverflow.com/a/826027/10805680 (remove 8 lines `typedef` at the bottom)
- ~~how about this https://github.com/robinrowe/libunistd/tree/master/unistd ???~~

## Benchmark

test video: length=40s, resolution=480×854

original pytorch implementation: `python inference_realesrgan_video.py` speed=1.8s/frame, vram=5.3gb

tensorrt (this repo): speed=1.5s/frame, vram=4.5gb

## 1️⃣ preparation

follow my guide to install Visual Studio + TensorRT: https://github.com/phineas-pta/NVIDIA-win/blob/main/NVIDIA-win.md

download opencv files https://github.com/opencv/opencv/releases then run `.exe` to unpack files (no installation)

download/clone this repo

inside, create those folders:
- `ckpt-pth/` to put `.pth` files
- `wts-file/` to put `.wts` files
- `trt-engine/` to put `.engine` files
- `samples/` input images
- `output/` output images

## 2️⃣ get models

✅ get any of these following checkpoints:
- https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
- https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth
- https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

⛔ anime checkpoints throw errors with tensorrt (maybe need edit c++ code):
- https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
- https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth

prepare a fresh env (venv/conda/virtualenv) then
```
pip install torch "torchvision<0.17" realesrgan --find-links https://download.pytorch.org/whl/torch_stable.html
```
convert pth to wts
```
python gen_wts.py -i ckpt-pth -o wts-file -n RealESRGAN_x4plus
```
change model name accordingly, possibly add `--fp16` for faster inference

## 3️⃣ build

edit file `real-esrgan.cpp`:
- line 9: only in case multiple gpu
- line 10: size 1 for 6gb vram, size 2 for 10gb vram, etc.
- line 13: fp32 or fp16
- lines 14-15: input image resolution
- line 16: number of color channels
- line 17: x2 or x4
- line 285: change output path if needed

⚠️ any change of those lines require rebuild

edit file `CMakeLists.txt`:
- line 11: path to OpenCV unpacked above
- line 12: cuda compute capability for e.g. 75 or 80 (this line can be removed)

need VS console

make new folder `build`
```
cmake -S . -B build
msbuild build\ALL_BUILD.vcxproj -noLogo -maxCpuCount -property:Configuration=Release
```
copy `███\OpenCV\build\x64\vc██\bin\opencv_world481.dll"` to `build\Release`

## 4️⃣ run

create images folder, for e.g. `samples`

⚠️ if input image resolution not match in file `real-esrgan.cpp` → complete garbage

normal cmd console (no need VS nor python)
```
set CUDA_MODULE_LOADING=LAZY

build\Release\real-esrgan -s wts-file\RealESRGAN_x4plus.wts trt-engine\RealESRGAN_x4plus.engine

build\Release\real-esrgan -d trt-engine\RealESRGAN_x4plus.engine samples
```

### upscale video

video-2-frame
```
ffmpeg -v warning -stats -hwaccel cuda -c:v vp9_cuvid -i input.webm "samples/%04d.png"
# change decoder accordingly: av1_cuvid - h264_cuvid - hevc_cuvid - vp9_cuvid
```
then run `real-esrgan`

⚠️ rebuild to match video resolution, if enough VRAM: increase `BATCH_SIZE` in `real-esrgan.cpp` line 10

frame-2-video (`*.png.png` coz weird things happen)
```
ffmpeg -v warning -stats -hwaccel cuda -i "output/%04d.png.png" -r 30 -c:v hevc_nvenc -pix_fmt yuv420p output.mp4
# change encoder accordingly: av1_nvenc - h264_nvenc - hevc_nvenc
# change frame rate accordingly
```
copy audio from input video? (to be tested)
```
ffmpeg -i output.mp4 -i input.webm -c:v copy -map 0:v:0 -map 1:a:0 outbis.mp4
```
