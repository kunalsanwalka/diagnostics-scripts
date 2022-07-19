
def make_genray_input(mnemonic="default",n_e0=3,n_eb=0.3,Te0=1000,Teb=100,Tfast0=25000,Ti0=100,\
                          rst=0.055,zst=0.4,phist=0,alfast=180,betast=-60,alpha1=2,alpha2=0,freq=110,mode="X",harmonic=1,\
                              nrays=10,ncones=1,beam_pow=1000000.0):

    default_file = "genray_template.in"
    directory = "/home/pizzo/genray_GUI/"
    #directory = "C:/Users/Jon Pizzo/Documents/UW Madison/Research/Genray GUI/"
    with open(directory+default_file,'r') as infile, open(directory+'genray.in','w') as outfile:
        for line in infile:
            if (line.find("mnemonic") != -1):
                line = "    mnemonic = '{}'\n".format(mnemonic)
            elif (line.find("dense0") != -1):
                line = "    dense0 = {:.2e}, {:.2e}, {:.2e}\n".format(n_e0*1E19,9*n_e0/10*1E19,n_e0/10*1E19)
            elif (line.find("denseb") != -1):
                line = "    denseb = {:.2e}, {:.2e}, {:.2e}\n".format(n_eb*1E19,9*n_eb/10*1E19,n_eb/10*1E19)
            elif (line.find("ate0") != -1):
                line = "    ate0 = {:.2f}, {:.2f}, {:.2f}\n".format(Te0/1000,Ti0/1000,Tfast0/1000)
            elif (line.find("ateb") != -1):
                line = "    ateb = {:.2f}, {:.2f}, {:.2f}\n".format(Teb/1000,Ti0/1000,Tfast0/1000) 
            elif (line.find("na1") != -1):
                line = "    na1 = {}\n".format(ncones)
            elif (line.find("na2") != -1):
                line = "    na2 = {}\n".format(nrays)
            elif (line.find("powtot") != -1):
                line = "    powtot = {:.1f}\n".format(beam_pow)
            elif (line.find("zst") != -1):
                line = "    zst = {:.3f}\n".format(zst)
            elif (line.find("rst") != -1):
                line = "    rst = {:.3f}\n".format(rst)
            elif (line.find("phist") != -1):
                line = "    phist = {:.3f}\n".format(phist)
            elif (line.find("alfast") != -1):
                line = "    alfast = {:.3f}\n".format(alfast)
            elif (line.find("betast") != -1):
                line = "    betast = {:.3f}\n".format(betast)
            elif (line.find("alpha1") != -1):
                line = "    alpha1 = {:.3f}\n".format(alpha1)
            elif (line.find("alpha2") != -1):
                line = "    alpha2 = {:.3f}\n".format(alpha2)
                
            elif (line.find("ioxm") != -1):
                if (mode == "X"):
                    line = "    ioxm = -1\n"
                else:
                    line = "    ioxm = 1\n"
            elif (line.find("frqncy") != -1):
                line = "    frqncy = {:.1f}\n".format(freq*1E9)
                    
            outfile.write(line)    
    return


if __name__ == "__main__":
    make_genray_input()
    print("Genray File Written")
