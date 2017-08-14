import os
import sys
import subprocess
import numpy as np
import scipy as sp
import scipy.stats as stat

def STATS(directoryname):
    """STATSStandAlone is used calculate the statistics from CL runs in a specific multi-run directory. 
	
	FOR THE STANDALONE: Need to run from the directory containing all the run files
	
		Arguments:
			
			directoryname - the name of the directory containing all the runs   

            Outputs:
    
    ******** Editions and Revisions **********
        Created 2017.08.14 by Nick Sherck
    
    
    """
    operatorname = str(operators.dat)
    
    statsOUTHam = []
    statsOUTStr = []
    statsOUTVol1 = []
    statsOUTVol2 = []
    
    DirHome = os.getcwd()
    DirPath = os.path.abspath(directoryname)
    for root, dirs, files in os.walk(directoryname): # walk through the run directories
        for i in dirs:
            fname = os.path.join(directoryname,i)
            for root2, dirs2, files2 in os.walk(fname):
                print fname
                for j in dirs2:
                    fname2 = os.path.join(fname,j)
                    os.chdir(fname2)
                    # Hamiltonian
                    column = decideColumn(operatorname,'Hamiltonian.Real')
                    warmup,Data,nwarmup = autoWarmupMSER(operatorname, column)
                    nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor = doStats(warmup,Data,'false','False','_{0}_{1}'.format(operatorname,column))
                    b = [mean,semcc]
                    statsOUTHam.append(b)
                    
                    # StressTensorConfined_VarConc1.Real
                    column = decideColumn(operatorname,'StressTensorConfined_VarConc1.Real')
                    warmup,Data,nwarmup = autoWarmupMSER(operatorname, column)
                    nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor = doStats(warmup,Data,'false','False','_{0}_{1}'.format(operatorname,column))
                    b = [mean,semcc]
                    statsOUTStr.append(b)
                    
                    # VolFrac1.Real
                    column = decideColumn(operatorname,'VolFrac1.Real')
                    warmup,Data,nwarmup = autoWarmupMSER(operatorname, column)
                    nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor = doStats(warmup,Data,'false','False','_{0}_{1}'.format(operatorname,column))
                    b = [mean,semcc]
                    statsOUTVol1.append(b)
                    
                    # VolFrac2.Real
                    column = decideColumn(operatorname,'VolFrac2.Real')
                    warmup,Data,nwarmup = autoWarmupMSER(operatorname, column)
                    nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor = doStats(warmup,Data,'false','False','_{0}_{1}'.format(operatorname,column))
                    b = [mean,semcc]
                    statsOUTVol2.append(b)
                    
                    os.chdir(DirHome)
	
    np.savetxt(directoryname+"statsOUTHam.txt",np.asarray(statsOUTHam,dtype=float))
    np.savetxt(directoryname+"statsOUTStr.txt",np.asarray(statsOUTStr,dtype=float))
    np.savetxt(directoryname+"statsOUTVol1.txt",np.asarray(statsOUTVol1,dtype=float))
    np.savetxt(directoryname+"statsOUTVol2.txt",np.asarray(statsOUTVol2,dtype=float))
    
	
	
def autoWarmupMSER(filehandler, col, debug=False):
    # Algorithm:
    # - Load all data
    # - Compute SEM with d samples truncated from beginning. The samples are supposed to be batch-averaged. This will be the case in most simulation data anyway.
    # - Find the value of d that minimizes SEM. This is the warmup length.
    filehandler = open(filehandler,'r')
    filehandler.seek(0)
    dummy, data = extractData(filehandler, col, 0)

    # MSER-5 - pre-block-average the data in chunks of 5 steps, and truncate integer counts of those steps
    m = 5 # Block size
    N = len(data)
    # Data will be padded in such a way that block average is not modified.
    # We are short by m-N%m points to make N/m an integer.
    # Then do the block averaging
    if N%m != 0:
        #dataint = np.pad(data, (0,m-N%m), mode='mean', stat_length=N%m)
        #databa = np.mean(dataint.reshape(int(N/m), m), axis=1)
        databa = np.mean(np.pad(data, (0,m-N%m), mode='mean', stat_length=N%m).reshape(int(N/m)+1, m), axis=1)
        N = N + m - N%m
    else:
        databa = np.mean(data.reshape(int(N/m), m), axis=1)

    if debug:
        fileout = open("debugMSER.dat","w")
    SEMlist=[]
    fullsize = len(databa)
    while True:
      # It is slow to do a full correlation-corrected statistical analysis in the inner loop
      # Rather than computing everything, just use correlation biased SEM
#      (n,(min,max),mean,semcc,kappa,unbiasedvar,autocor)=doStats(dummy,databa,False,False)
      sem = stat.sem(databa)
#      var = unbiasedvar * (n-1)/n # The MSER method uses the biased sample variance as a measure of homogeneity of the data
#      VARlist.append(var)
#      SEMlist.append(semcc)
      SEMlist.append(sem)
      if debug:
          fileout.write("{0} {1}\n".format(len(databa),semcc))
      # Ensure we don't go below 25% of data remaining or 5 samples or that we don't iterate more than 1000 times.
      Nrem = len(databa)
      if Nrem <= 0.25*fullsize or Nrem < 5:
          break
      # TO DO: replace with masked array to avoid repeated copy
      databa = databa[1:] # Delete the first m elements

    if debug:
        fileout.close()

    # Find the index that minimizes the variance
    idx = np.argmin(SEMlist) # idx is therefore the number of warmup blocks to be removed; m*idx is # samples
    idx = idx*m

    if idx > 0:
        warmupdata = data[:idx-1]
    else:
        warmupdata = np.empty([0])
    proddata   = data[idx:]

    return warmupdata, proddata, idx	

def decideColumn(filehandler,observablename):
    # Be sure we're at the beginning of the file
	filehandler = open(filehandler,'r')
	filehandler.seek(0)
    # Get the header line - in case we want to choose column based on string operator name
    # Procedure: keep reading lines until the last header line. If we are choosing a column
    # based on a label, check for it on the last header line. If we're choosing based on an index,
    # read the first non-comment line to check the allowed column index
	line = filehandler.readline()
	if len(line)==0:
		sys.stderr.write("Error: first line of file is empty\n")
		sys.exit(1)	
	cols = line.split()
	headerfound=False
	while cols[0]=="#":
		headerfound=True
		headercols=cols
		line = filehandler.readline()
		cols = line.split()
		if len(line) == 0:
			break
	if headerfound:
		# Set dataset column from header label instead of column index
		if not(observablename in headercols):
			sys.stderr.write("\nError: observable name not in list\n")
			sys.exit(1)
		return headercols.index(observablename)-1 # Subtract 1 because of # marker
	else:
		sys.stderr.write("\nError: instruction to detect column from observable name. No valid header in file.\n")
		sys.exit(1)

def extractData(filehandler,column,warmup):
    # Be sure we're at the beginning of the file
    filehandler.seek(0)
    # Read the file - First get all of the data
    AllData=np.loadtxt(filehandler)
    # Copy to a 1D Numpy array for faster analysis
    Data1=AllData[:,column]
    # Check for NaN in the data
    if np.isnan(np.min(Data1)): # Note: min() propagates NaN. Checking result of reduction operator faster than creating a vector of results.
        sys.stderr.write("WARNING: NaN in data. Run diverged. Proceeding with bad samples removed.\n")
        Data1=Data1[numpy.isfinite(Data)]
    # Check data size > warmup
    if Data1.size < warmup:
        sys.stderr.write("WARNING: Warmup length is greater than sample size. Reducing to warmup=0\n")
        warmup=0
    # Return warmup and production arrays
    if warmup > 0:
        return Data1[:warmup-1],Data1[warmup:]
    else:
        return np.empty([0]),Data1


def doStats(warmupdata,Data,doGraphs=False,doWriteStdout=False,graphFilenameStub=''):
    # Mean, min, max, variance
    (nsamples,(min,max),mean,unbiasedvar,skew,kurtosis)=stat.describe(Data) # Unbiasedvar is actually the reduced-bias estimator of population variance (~1/(N-1)...)
    # Standard error of mean:
    sem=stat.sem(Data) # Not yet correlation corrected

    # Compute the autocorrelation function:
    Data_dcshift=Data-mean
    #DataNorm=np.sum(np.square(Data_dcshift))
    cor=np.correlate(Data_dcshift,Data_dcshift,mode='same')/unbiasedvar
    autocor=cor[cor.size/2:]
    autocor=autocor/np.arange(nsamples-1,nsamples-1-autocor.size,-1) # Note -1 for 0-based indexing

    # Choose where to cutoff the autocorrelation time sum
    cutoff=autocor.size
    j=0
    while j<cutoff:
        if autocor[j] < np.sqrt(2./(nsamples-j)):
            cutoff = np.minimum(cutoff,5*j)
        j=j+1
    # Compute correlation time
    kappa=1.+2.*np.sum(autocor[1:int(2.*cutoff/5.)])
    # We can also make an array of all possible cutoffs
    if doGraphs:
      kappa_cutoffdep=np.ones(autocor.size)
      for jc in range(1,autocor.size):
        kappa_cutoffdep[jc]=1+2*np.sum(autocor[1:jc])
    # Update the standard error of the mean for a correlation correction
    semcc=sem*np.sqrt(kappa)

    # Manual (non-Numpy) autocorrelation function for transparency - verified equal
    #j=0
    #cutoff=nsamples
    #autocor_m=np.zeros(cutoff)
    #while j < cutoff:
    #    autocor_m[j]=0.
    #    for i in range(0,nsamples-j):
    #        autocor_m[j] = autocor_m[j] + (Data[warmup+i]-mean)*(Data[warmup+i+j]-mean)
    #    autocor_m[j]=autocor_m[j]/(unbiasedvar*(nsamples-j))
    #    if autocor_m[j] < np.sqrt(2./(nsamples-j)):
    #        cutoff = np.minimum(cutoff,5*j)
    #    j=j+1

    if doWriteStdout == True:
        print "  - Mean                    = ",mean," +/- ",semcc
        print "  - Equilibrated samples    = ",nsamples
        print "  - Correlation time        = ",kappa
        print "  - Effective # samples     = ",nsamples/kappa
        print "  - Reduced-bias variance   = ",unbiasedvar
        # note that there is no unbiased estimator for the population standard deviation. We can use sqrt(var) as a indicative estimator.
        print "  - S.D. (unbiased, biased) = ",np.sqrt(unbiasedvar),np.std(Data,ddof=0) # ddof is correction to 1/N...using ddof=1 returns regular reduced-bias estimator
        print "  - Skewness                = ",skew
        print "  - Kurtosis                = ",kurtosis
        print "  - Min, Max                = ",min,max
        print  # Reduced bias estimator - test vs. above from sqrt(var)

    if doGraphs:
        import matplotlib.pyplot as pl # If we import pylab instead, we get matplotlib.pyplot and numpy both under the global namespace for MATLAB-like syntax and reduced typing (useful for interactive use)

        # Plot some things
        pl.figure(num=1,figsize=(15,10))
        #
        pl.subplot(221) # Select first panel in 2x2 grid...
        pl.title("Trace of Data")
        pl.plot(np.concatenate([warmupdata,Data]))
        pl.ylim([0.98*min,1.02*max])
        pl.axhline(mean,color='red')
        pl.axvspan(0,len(warmupdata),color='green',alpha=0.5)
        pl.axvline(len(warmupdata),color='green')
        pl.xlabel("Sample index")
        #
        # Generate a histogram of the data
        # Not needed now - just do a histogram plot directly
        #hist=stat.histogram(Data)
        #print hist
        #hist=np.histogram(Data)
        #print hist[0]
        #print hist[1]
        #print type(hist[0])
        pl.subplot(222)
        pl.title("Histogram of Data")
        n,bins,patches=pl.hist(Data,25,normed=1,facecolor="green",alpha=0.5)
        #ygauss=stat.norm(bins,mean,np.sqrt(unbiasedvar))
        pl.plot(bins,stat.norm.pdf(bins,mean,np.sqrt(unbiasedvar)),'r--')
        #
        pl.subplot(223)
        pl.plot(autocor[:cutoff])
        x=np.arange(0,cutoff)
        pl.plot(x,np.exp(-x/kappa))
        pl.title("Autocorrelation function")
        pl.xlabel('$\\tau$')
        pl.ylabel('$C\\left(\\tau\\right)$')
        pl.axhline(0,color='black')
        #pl.xlim(0,plotxmax)
        #
        pl.subplot(224)
        pl.plot(kappa_cutoffdep)
        pl.title("Correlation time estimator vs. cutoff")
        #pl.xlabel('$\\tau_{cut}$')
        #pl.ylabel('$\\Kappa$')
        #pl.axhline(0,color='black')
        #
        pl.savefig("stats_{0}.png".format(graphFilenameStub))

    return (nsamples,(min,max),mean,semcc,kappa,unbiasedvar,autocor)
	
	
if __name__ == "__main__":
# For command-line runs, build the relevant parser
	import argparse as ap
	parser = ap.ArgumentParser(description='Outputs the statistics from Hamiltonian, StressTensor1, VolFrac1 and VolFrac2')
	parser.add_argument('-f','--file',type=str,help='Filename containing operators.dat')
	args=parser.parse_args()
	STATS(args.file)
	