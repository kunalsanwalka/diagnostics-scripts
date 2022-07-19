import run_fast_ion_gui

def orbit_integrator(zgrid,rgrid,bz,br):
    run_fast_ion_gui.pass_field_and_run_gui(rgrid,zgrid,br,bz)
