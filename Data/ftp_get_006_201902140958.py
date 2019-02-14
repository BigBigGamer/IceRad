#
# PPS Python ftp script
#
# Usage:  python <script name>
#

import os, sys, re, hashlib
from getpass import getpass
from ftplib import FTP

downloadCount=0
skipCount=0

def hashfile(filename, blocksize=65536):
    hasher = hashlib.sha1()
    localfile=open(filename, 'rb')
    buf = localfile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = localfile.read(blocksize)
    localfile.close()
    return hasher.hexdigest()

def getFile(filepath,cksum=None):
    global downloadCount,skipCount
    path,filename=os.path.split(filepath)
    connection.cwd(path)
    download=True
    ftpSize=connection.size (filename)
    # determine if file exists in local directory
    if (os.path.exists(filename)):
        # check size and cksum
        if (cksum):
            sha = hashfile(filename)
            if (cksum == sha):
                download=False
        else:
            filesize=os.path.getsize(filename)
            if (ftpSize==filesize):
                download=False

    if (download):
        # if not exists or file checks do not match, get file.
        downloadCount+=1
        sys.stdout.write( str(downloadCount)+') Downloading '+filename+'   '+str(ftpSize)+' bytes  ')
        sys.stdout.flush()
        localfile=open(filename, 'wb')
        connection.retrbinary('RETR ' + filename, localfile.write, 1024)
        localfile.close()

        if (cksum):
            sys.stdout.write('cksum ')
            sys.stdout.flush()
            sha = hashfile(filename)
            if (cksum == sha):
                print ('Pass')
            else:
                print ('FAIL')
        else:
            print ('Done')
    else:
            print ('Already downloaded '+filename)
            skipCount+=1


if __name__ == '__main__':
    userinfo=('aps98@mail.ru','aps98@mail.ru')
    print ('Connecting to PPS')
    connection = FTP('arthurhou.pps.eosdis.nasa.gov')
    print (connection.getwelcome())
    connection.login(*userinfo)
    connection.sendcmd('TYPE i')

#
# The following is the list of PPS files to transfer:
#
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180303-S124914-E125447.022789.V06A.HDF5','1bbefb463aba0298ae6774e9145c92c006874871')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180303-S142206-E143033.022790.V06A.HDF5','ea09a29ded93965513ec8ff6a9e2fef7d4384f74')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180303-S160019-E160356.022791.V06A.HDF5','4aa6b9b7e55b8bcc66a52cb6939108460f940164')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180303-S173613-E174122.022792.V06A.HDF5','2e55bf0b8488971cf8b680feb92d267cf5f796ed')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180303-S191152-E191819.022793.V06A.HDF5','7d5c1f6200b6a02cf1efb094c79acaefe4b2cfba')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180303-S204653-E205403.022794.V06A.HDF5','a1eb2af979e02b9795e887abaffa6ea1da9a1496')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180303-S222403-E222636.022795.V06A.HDF5','5d3f822a13c8c592054582237ee6a5c294e66fdd')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180304-S115730-E120116.022804.V06A.HDF5','a1978f3151b87ad045423921c4bfa7938edab86c')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180304-S133003-E133752.022805.V06A.HDF5','12ba8c01dc32485dcdce8a588120169a3455bb57')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180304-S150654-E151213.022806.V06A.HDF5','73797e331d8a72f6b9074aca0a81e6a28cc19492')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180304-S164322-E164828.022807.V06A.HDF5','2f47eae2e19017e92254ba1c4d5fd8763c95e99d')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180304-S182008-E182439.022808.V06A.HDF5','e17f9781588f8d396cff7301f7a6b40f65e4ecab')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180304-S195405-E200219.022809.V06A.HDF5','57a4b795cd1a37db8ec6e688a0628321d10c2105')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180304-S213015-E213453.022810.V06A.HDF5','c913f08c93660cd7b518cf135da2b13d18c9013a')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180305-S110547-E110710.022819.V06A.HDF5','2b6094861b34a4061e576c13f7b1582adf8802d4')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180305-S123820-E124454.022820.V06A.HDF5','d1cd435e720d20809e0054d2d6286fbabd6d739f')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180305-S141300-E142020.022821.V06A.HDF5','d251a51fa593981330b58909fafebde9dedd2665')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180305-S155024-E155301.022822.V06A.HDF5','50f14942aaef612ab4e531e1a62308d43700ae68')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180305-S172825-E173123.022823.V06A.HDF5','95264a1179281ed86c90b2c14ea9aa865d781e3d')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180305-S190122-E190906.022824.V06A.HDF5','583035c779860bbbf087f28ee84d426f24dfaac5')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180305-S203654-E204308.022825.V06A.HDF5','2fa40df82644c428ca25f64d68cec82af795c823')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180305-S221501-E221541.022826.V06A.HDF5','7284e18f7d54dde398764bba8f1483e6f9edc8b9')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180306-S114636-E115139.022835.V06A.HDF5','2e590d0086c78d00f04cb3c1c9f75d068c987260')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180306-S131911-E132737.022836.V06A.HDF5','d3b707c5f8897c69378b7c9daf805c787a10a3e1')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180306-S145711-E150118.022837.V06A.HDF5','d2c2d3fed29bc7516a112b6620bae441d9b2d027')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180306-S163315-E163821.022838.V06A.HDF5','9c84782cfa575884a4de50e31d3118f25a53596a')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180306-S180913-E181501.022839.V06A.HDF5','ec3171a11e92e5d72216e1d8abbfe3c62c39e8cd')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180306-S194353-E195124.022840.V06A.HDF5','25c286eac38f61b2437becaad33ff7fe32c7c0e2')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180306-S212042-E212357.022841.V06A.HDF5','7b7e48ffbf011ec45d2631b7a34743f7efca836f')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180307-S105451-E105758.022850.V06A.HDF5','a25d942f1e3acd10717dd28e8be1b4fc7bcb52c5')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180307-S122724-E123452.022851.V06A.HDF5','925fc20828db57f7e9ce048005d00acbef5f0fc4')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180307-S140339-E140938.022852.V06A.HDF5','8c572aca9941064ecc12537319db2e37bafda413')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180307-S154024-E154530.022853.V06A.HDF5','fa76db700a8ab56fc3614588e4921f173ae1bcd0')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180307-S171733-E172135.022854.V06A.HDF5','fe070fdddb2f9eb0ca709afc07899515283c913b')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180307-S185114-E185941.022855.V06A.HDF5','e0fbbcada0d21d71ccef5ac57e2ee07ce6121e81')
    getFile('/gpmuser/aps98@mail.ru/pgs/2A-CS-127E63N168E42N.GPM.Ku.V8-20180723.20180307-S202713-E203223.022856.V06A.HDF5','584ecc1b45df474f7f6e83f18441689156116ab3')
#
# Transfer complete; close connection
#
    connection.quit()
    print ('Number of files downloaded: '+str(downloadCount))
    if (skipCount>0):
        print ('Number of files already downloaded: '+str(skipCount))
