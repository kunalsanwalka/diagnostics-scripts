from pleiades import Device, RectangularCoil


class WHAM_VNS(Device):
    """The Device object representing the Wisconsin HTS Axisymmetric Mirror Volumetric Neutron Source

    Attributes
    ----------
    hts1 : RectangularCoil object
        A coil for the positive Z HTS mirror coil
    hts2 : RectangularCoil object
        A coil for the negative Z HTS mirror coil
    """

    def __init__(self):
        # Global default patch settings
        super().__init__()

        # Set HTS mirror coil default parameters
        dr = 0.0255
        dz = 0.04
        nr = 12
        nz = 6
        r_min = 0.3
        z0 = 1.25
        r0 = r_min + dr*nr/2

        self.hts1 = RectangularCoil(r0, z0, nr=nr, nz=nz, dr=dr, dz=dz)
        self.hts2 = RectangularCoil(r0, -z0, nr=nr, nz=nz, dr=dr, dz=dz)
        self.hts1.current = 247953
        self.hts2.current = 247953
