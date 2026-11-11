# Índice de resultados

## Ejecuciones

- [20261105T112852Z_wave_2d_sine_liesolver_s0_2c12a8b0](20261105T112852Z_wave_2d_sine_liesolver_s0_2c12a8b0/): preliminar, error L2 0.000399385531, tiempo 0.386139683 s.
- [20261105T112907Z_wave_2d_sine_classical_s0_691acfc8](20261105T112907Z_wave_2d_sine_classical_s0_691acfc8/): preliminar, error L2 6.22740327e-05, tiempo 0.023015866 s.
- [20261105T113145Z_wave_2d_sine_4pi_liesolver_s0_100474e7](20261105T113145Z_wave_2d_sine_4pi_liesolver_s0_100474e7/): **informe final**, error L2 0.000399725199, tiempo 0.200240952 s.
- [20261105T113146Z_wave_2d_sine_4pi_classical_s0_f3dcad57](20261105T113146Z_wave_2d_sine_4pi_classical_s0_f3dcad57/): **informe final**, error L2 0.000612071574, tiempo 0.132252764 s.
- [20261110T170540Z_wave_3d_sine_4pi_liesolver_s0_4fbe0ec2](20261110T170540Z_wave_3d_sine_4pi_liesolver_s0_4fbe0ec2/): preliminar, error L2 0.00082340533, tiempo 164.048065 s.
- [20261110T170554Z_wave_3d_sine_4pi_classical_s0_b6d4e3a7](20261110T170554Z_wave_3d_sine_4pi_classical_s0_b6d4e3a7/): preliminar, error L2 0.00140435949, tiempo 62.8004126 s.
- [20261110T170934Z_wave_3d_sine_4pi_liesolver_s0_06fe499d](20261110T170934Z_wave_3d_sine_4pi_liesolver_s0_06fe499d/): **informe final**, error L2 0.00082340533, tiempo 134.64685 s.
- [20261110T171153Z_wave_3d_sine_4pi_classical_s0_f53a6918](20261110T171153Z_wave_3d_sine_4pi_classical_s0_f53a6918/): **informe final**, error L2 0.00140435949, tiempo 60.8931881 s.

## Comparaciones

- [2D preliminar, modo pi](comparison_ceb87c04a4a5/).
- [2D final, modo 4pi](comparison_cc9db5edbc91/).
- [3D final, modo 4pi](comparison_b018ba5385a6/).

## Archivos de cada ejecución

`config.yaml` y `metadata.json` registran configuración, semilla, versiones y tiempos. `metrics.json` contiene las métricas; `history.csv` el entrenamiento/integración; `constraints.csv` las condiciones con objetivo/predicción/error; `domain.npz` las coordenadas y campos de referencia/predicción/error; `slices.csv` y `error_by_time.csv` los números de las figuras. Los modelos/checkpoints NPZ permiten recarga. Las figuras PNG se generan a partir de los datos exportados.

En `domain.npz`, las columnas son `(x,y,t)` o `(x,y,z,t)`; la forma de la malla es `(Nt,Ny,Nx)` o `(Nt,Nz,Ny,Nx)`. Consultar las claves del archivo con `numpy.load(path).files`. Las comparaciones incluyen CSV y NPZ con los números representados.

El [informe combinado](report_wave_2d_3d_20261110_2b9eddf1/report.tex) utiliza únicamente las cuatro ejecuciones marcadas como finales.
