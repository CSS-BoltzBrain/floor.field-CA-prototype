Go to `floor_field_ca` and ivoke this command to have a quick try:

```
python main.py --config configs/supermarket.yaml --gif
```

or

```
python main.py --config configs/train_cabin.yaml
```


Take a look of specification at `specification/specification.md` to have an overview of the software architechure. In short:
   - a configrable map via `floor_field_ca/configs/`. See `floor_field_ca/configs/*.yaml` for example.
   - an agent class: so we can add/remove traits of agents easily. (see `class Agent`)
   - an export module to help us export the state from each simulation state. This will greately help us when analyzing and recognizing complex behavior (this is a lesson from my boids assignment)

At this moment, you do not need to read specification.md carefully. Just an overview about what I suppose our project should look like is sufficient enough for our discussion during practical section.
I will appreciate your input regarding this perspective.


I am still working on how to recognize and proof complex behavior from the system. Also, I am not very sure if mixing CA and agent-based simulation is a good idea. Still exploring that as well.


