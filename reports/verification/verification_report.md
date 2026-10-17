# PROJECT 0 Verification Report

- Generated (UTC): 2026-10-17T10:49:38.941696+00:00
- Status: **PASS**
- Checks passed: 6/6

## Scope and limitations

This report covers software and numerical verification under the
specific test conditions described below. It does not validate
the model against experimental flight data or certify a vehicle.

## analytical_free_fall: PASS

```json
{
  "position_error_m": 0.0,
  "velocity_error_m_s": 1.4210854715202004e-14,
  "duration_s": 1.0
}
```

## coordinate_frame_rotation: PASS

```json
{
  "orthogonality_error": 1.4353152422874004e-17,
  "determinant": 1.0,
  "round_trip_error": 4.440892098500626e-16
}
```

## timestep_convergence: PASS

```json
{
  "reference_dt_s": 0.000625,
  "errors_against_fine_reference": {
    "0.04": 3.370713072873021e-10,
    "0.02": 2.3649537305268144e-11,
    "0.01": 5.166647388695313e-12,
    "0.005": 4.595401841008586e-12
  },
  "interpretation": "Errors are measured against a finer numerical run, not experimental data."
}
```

## torque_free_rotation: PASS

```json
{
  "relative_rotational_energy_drift": 3.3072638222105947e-15,
  "relative_body_angular_momentum_norm_drift": 1.5514641386586038e-15,
  "duration_s": 2.0,
  "dt_s": 0.001
}
```

## quaternion_normalisation: PASS

```json
{
  "maximum_norm_error": 0.0
}
```

## deterministic_reproducibility: PASS

```json
{
  "samples": 101,
  "bitwise_identical_outputs": true
}
```
