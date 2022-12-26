# INSTALL

## Linux (Fedora)
```
dnf install -y \
  tesseract-devel \
  leptonica-devel \
  python3-opencv \
  python3-devel \
  android-tools

pip install -r requirements.txt

cd calc-odds/
go build .

# scrcpy (optional)
dnf copr enable zeno/scrcpy
dnf install -y scrcpy
```

## Windows
```
conda create --name poker python=3.9
conda activate poker
conda install -c simonflueckiger tesserocr
conda install opencv
pip install -r requirements-windows.txt
cd .\calc-odds\
go build .

# install adb
# extract https://dl.google.com/android/repository/platform-tools-latest-windows.zip to `c:\platform-tools` and add to PATH
adb devices

# scrcpy (optional)
pip install scrcpy
```

# Usage
```
adb devices
python pokerface.py
```
