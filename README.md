**Combinatorial Decision Making and Optimization**
# Multiple Couriers Problem

Instructions to run the models:
- On Unix run `./run.sh <TECHNIQUE> <INSTANCE>`
- On Windows run `.\run.bat <TECHNIQUE> <INSTANCE>`
- `<TECHNIQUE>` can be `cp`, `smt`, `mip` or `all`
- `<INSTANCE>` can be a single instance number (ex. `1`), a range (ex. `4-8` for instances 4 to 8 included) or `all` to run all instances 
- If you prefer raw Docker, build the image with `docker build . -t cdmo` and then launch it with `docker run --mount type=bind,src=./res,dst=/app/res cdmo /venv/bin/python /app/run.py <TECHNIQUE> <INSTANCE>`