# System A — Mechanism (DISCLOSED)

System A is the **worked reference**. Its mechanism is given so you can build and validate
your whole workflow against a known answer before pointing it at the blind System B. If your
method cannot recover System A, it will not recover anything.

System A is a **different reaction** from System B — do not assume the blind answer looks
like this one.

## Network — model M1

Two parallel bimolecular routes from the same reactants:

```
R1:  A + B  ->  P      (main product)
R2:  A + B  ->  Q      (impurity, parallel to product)
```

Both steps are elementary: first order in each reactant, second order overall.

## Rate law

Each rate constant is Arrhenius, written in the same convention the candidate library uses:
its value at a **reference temperature of 75 °C** together with an activation energy.

```
k_j(T) = k_j(75 °C) * exp( -(Ea_j / R) * (1/T - 1/T_ref) ),   T_ref = 348.15 K, R = 8.314 J/mol/K
d[A]/dt = d[B]/dt = -(k1 + k2)·[A][B]
d[P]/dt = k1·[A][B]
d[Q]/dt = k2·[A][B]
```

Concentrations are in mol/L, time in minutes.

## What is measured

All four species **A, B, P, Q** are quantified, across five sets of conditions varying
temperature and initial concentrations, with complete concentration–time profiles. Noise is
2 % (relative) on the main components and 5 % on the trace species Q. No values are censored.

Parameter values (the rate constants and activation energies) are **not** given — recovering
them from the data is the point of the reference exercise.

See `../README.md` for the file schema and units.
