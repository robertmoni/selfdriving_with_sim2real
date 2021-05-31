# self_driving_with_sim2real


# 1. Docker init  

```nvidia-docker run --rm -v /home/robertmoni/projects/selfdriving_with_sim2real:/home/selfdriving_with_sim2real -td -p 2249:22 -p 7080:6006 -p 7081:8888 -w /home/general/  --name dockerrm rmc26/selfdriving_with_sim2real```

# 2. Jupyter

```jupyter notebook --no-browser  --port 8805 --ip 0.0.0.0```
