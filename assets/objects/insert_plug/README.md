# Supplied STL objects

The scene displays the original `assets/放置桶.STL` twice and the original
`assets/螺栓.STL` once. Neither source file is changed. STL does not carry units;
their coordinates are interpreted as millimetres and scaled by `0.001` for
MuJoCo. Both original meshes are watertight and have their long axis along +Y.

| Measurement | Holder | Bolt |
| --- | --- | --- |
| Original XYZ bounds, mm | `[0, 0.0482900068, 0]` to `[25, 40.0482902527, 25]` | `[0, 0, 0]` to `[27.7128124237, 37.5, 24]` |
| Upright height | 40 mm | 37.5 mm |
| Bore / shaft nominal diameter | 11 mm | 10 mm |
| Cavity floor / shaft end below head | 5 mm above holder bottom | 30 mm above bolt tip |
| Cavity depth / head height | 35 mm | 7.5 mm |
| Exterior / head | Fluted upper barrel, 20 mm maximum diameter; 25 mm base width | Hexagonal, 24 mm across flats, 27.7128 mm across corners |

The bolt shaft has a rounded first millimetre and no modeled thread. Its head
rests on the holder rim: a seated tip is 10 mm above the holder bottom, leaving
5 mm between the tip and the actual cavity floor. The grasp centre is 3.75 mm
above the rim. This is a straight extraction/insertion task, not screw driving.

For both visuals, use quaternion `[sqrt(0.5), sqrt(0.5), 0, 0]` (MuJoCo WXYZ),
which rotates +90 degrees about X. After the `0.001` mesh scale, translation is
`[-0.0125, 0.0125, -0.0000482900068164]` for the holder and
`[-0.013856406211853, 0.012, 0]` for the bolt. Holder body placement then adds
the configured socket XY and bottom Z. The bolt body origin is its lower tip.

## Collision geometry and limitations

`collision.json` contains source SHA-256 hashes and convex vertices/faces in
metres. The scene verifies source hashes and transforms before using it.
Original meshes are visual-only. No convex hull of the entire hollow holder
is used for contact.

Each holder uses 266 convex parts. The source bore has 38 top-rim vertices;
radial sector boundaries follow these vertices. Separate height bands cover
the solid 0–5 mm floor, each millimetre of the 5–10 mm flared foot, and the
10–40 mm barrel. Hulls are formed from original triangles clipped to each
sector and height band. This retains the source bore facets, actual floor,
and top entrance plane. Outer flutes and the curved foot are locally
convexified. The two holder instances share the same 266 mesh assets.

The holder collision volume is **0.6277% larger** than the original solid.
A geometric check sampled 720 azimuths at 19 heights (13,680 radial rays).
The sampled bore discrepancy was below `6e-13 m`; no sampled radial solid
undercoverage was found. The largest sampled outward radial expansion was
**0.718 mm**, at the bottom edge of the fluted foot, 0.025 mm above the base.
These are sampled measurements, not guaranteed global error bounds. The foot
approximation is away from the bore and head support used by this task.

The bolt has two contact meshes: its exact convex hexagonal head and a convex
hull of the shaft, retaining its rounded tip. The original shaft triangulation
is slightly nonconvex; the combined contact volume is **0.00383% larger** than
the original bolt. Mass is distributed between the two contact pieces in
proportion to their volumes, using the configured total mass.

Regenerate the manifest from the untouched source STLs with the shared
environment:

```bash
env -u PYTHONPATH conda run -n mujoco python -B scripts/prepare_insert_plug_objects.py
```

The generator needs only the existing NumPy, SciPy and trimesh dependencies;
scene construction reads the precomputed JSON and does not run decomposition.
