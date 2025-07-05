**Combinatorial Decision Making and Optimization**
# Multiple Couriers Problem

Instructions to run the Docker image:

1. Build it with `docker compose build`
   - If you prefer raw Docker use `docker build . -t cdmo`
2. Run it with `docker compose run cdmo /app/run.sh <TECHNIQUE> <INSTANCE>`
   - If you prefer raw Docker use `docker run --rm cdmo /app/run.sh <TECHNIQUE> <INSTANCE>`, linking a volume to `/app/res/` if you want to export the results
   - `<TECHNIQUE>` can be `cp`, `smt`, `mip` or `all`
   - `<INSTANCE>` can be a single instance number (ex. `1`), a range (ex. `4-8` for instances 4 to 8 included) or `all` to run all instances 