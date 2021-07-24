import pyvirtualdisplay
import subprocess
import os

_display = pyvirtualdisplay.Display(visible=False,  # use False with Xvfb
                                    size=(1400, 900))
_ = _display.start()
print("DISPLAY == ", os.environ.get("DISPLAY"))

# subprocess.run(["echo", "$DISPLAY"])