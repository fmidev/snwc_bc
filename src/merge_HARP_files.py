import sys,os,getopt
import sqlite3
import pandas as pd

def merge(path):
    res=[]
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            try:
                db = sqlite3.connect(f)
                res.append(pd.read_sql_query("select * from FC",db))
            except:
                continue
    return pd.concat(res)
    
def output(data,outfile):
    db = sqlite3.connect(outfile)
    data.to_sql("FC",db)
    
def main():
    # Fetch command line arguments
    #------------------------------------------------------------------------------------------------------------
    options, remainder = getopt.getopt(sys.argv[1:],[],['path=','outfile=','help'])
    
    for opt, arg in options:
        if opt == '--help':
            print('merge_HARP.py path=<path> outfile=<outfile>')
            exit()
        elif opt == '--path':
                path = arg
        elif opt == '--outfile':
                outfile = arg

    # Exit with error message if not all command line arguments are specified
    try:
        path, outfile
    except NameError:
        print('ERROR! Not all input parameters specified: ')
        exit()
    #------------------------------------------------------------------------------------------------------------​
    # Execute program
    #------------------------------------------------------------------------------------------------------------
    tab = merge(path)
    output(tab,outfile)
    #------------------------------------------------------------------------------------------------------------
    # ​
if __name__ == "__main__":
    main()
