from pleiades import Device, RectangularCoil


class WHAM(Device):
    """The Device object representing the Wisconsin HTS Axisymmetric Mirror.

    Attributes
    ----------
    hts1 : RectangularCoil object
        A coil for the positive Z HTS mirror coil
    hts2 : RectangularCoil object
        A coil for the negative Z HTS mirror coil
    w7as1 : RectangularCoil object
        A coil for the positive central cell coil
    w7as1 : RectangularCoil object
        A coil for the negative central cell coil
    """

    def __init__(self):
        # Global default patch settings
        super().__init__()

        # Set HTS mirror coil default parameters
        z0 = 0.98
        dr, dz = 0.01377, 0.01229
        nr, nz = 23, 6
        r0 = 0.06+dr*nr/2
        self.hts1 = RectangularCoil(r0, z0, nr=nr, nz=nz, dr=dr, dz=dz)
        self.hts2 = RectangularCoil(r0, -z0, nr=nr, nz=nz, dr=dr, dz=dz)
        self.hts1.current = 35140
        self.hts2.current = 35140

        z0 = 0.265
        dr, dz = 0.4/12, 0.183/2
        nr, nz = 12, 2
        r0 = 0.55 + dr*nr/2
        self.w7as1 = RectangularCoil(r0, z0, nr=nr, nz=nz, dr=dr, dz=dz)
        self.w7as2 = RectangularCoil(r0, -z0, nr=nr, nz=nz, dr=dr, dz=dz)
        self.w7as1.current = 26000
        self.w7as2.current = 26000




