function k1=sortAll(LaKu,LoKu,La01,Lo01,La02,Lo02,...
    d21,d22,j11,j22,dirn,whatK,Lsw,kmPerSquare,...
    sizeKu,IncAngleKu,sigmaKu,preciprateKu,secofdayKu)

k1=1;
R = deg2km(distance(La01,Lo01,La02,Lo02))*0.5;
if (d22<Lsw && d21<Lsw)|| (d22>Lsw && d21<R) || (d22<R && d21>Lsw)
    jav=floor((j11+j22)/2);
    
    jstku = jav-floor(R/kmPerSquare)-4;
    jfinku = jav+floor(R/kmPerSquare)+4;
    if jstku<1
        jstku = 1;
    end
    if jfinku>sizeKu(2)
        jfinku = sizeKu(2);
    end
    % cut Ku
    k = 1;
    jj1=zeros(sizeKu(1)*(jfinku-jstku),1);
    for i1 = 1:sizeKu(1)
        for j1 = jstku:jfinku
            if LaKu(i1,j1)<=La02 &&  LaKu(i1,j1)>=La01 &&  LoKu(i1,j1)<=Lo02 &&  LoKu(i1,j1)>=Lo01
                if k==1
                    [folderStatus,folderMessage]=mkdir(dirn);
                    fidLa = fopen(strcat(dirn,'\La',whatK,'.txt'),'wt');
                    fidLo = fopen(strcat(dirn,'\Lo',whatK,'.txt'),'wt');
                    fidSig = fopen(strcat(dirn,'\Sig',whatK,'.txt'),'wt');
                    fidInc = fopen(strcat(dirn,'\Inc',whatK,'.txt'),'wt');
                    fidTime = fopen(strcat(dirn,'\Sec',whatK,'.txt'),'wt');
                    fidPrec = fopen(strcat(dirn,'\Prec',whatK,'.txt'),'wt');
                    fidK = fopen(strcat(dirn,'\area',whatK,'.txt'),'wt');
                end
                k1=2;
                jj1(k) = j1;
                k = k+1;
            end
        end
    end
    jj1=jj1(1:k-1);
    nmin=min(jj1);
    nmax=max(jj1);
        
    for ii=1:sizeKu(1)
        for jj=nmin:nmax
            
            fprintf(fidK,'%e   %e   %e   %e   %e   %e   \r\n',LaKu(ii,jj),LoKu(ii,jj),IncAngleKu(ii,jj),sigmaKu(ii,jj),preciprateKu(ii,jj),secofdayKu(jj,1));
            
        end
    end
    
    dlmwrite(strcat(dirn,'\Inc',whatK,'.txt'),IncAngleKu(:,nmin:nmax),'delimiter','\t');
    dlmwrite(strcat(dirn,'\Sig',whatK,'.txt'),sigmaKu(:,nmin:nmax),'delimiter','\t');
    dlmwrite(strcat(dirn,'\La',whatK,'.txt'),LaKu(:,nmin:nmax),'delimiter','\t');
    dlmwrite(strcat(dirn,'\Lo',whatK,'.txt'),LoKu(:,nmin:nmax),'delimiter','\t');
    dlmwrite(strcat(dirn,'\Prec',whatK,'.txt'),preciprateKu(:,nmin:nmax),'delimiter','\t');
    dlmwrite(strcat(dirn,'\Sec',whatK,'.txt'),secofdayKu(nmin:nmax),'delimiter','\t');
    
    if k>1
        fclose(fidK);
        fclose(fidLa);
        fclose(fidLo);
        fclose(fidSig);
        fclose(fidInc);
        fclose(fidTime);
        fclose(fidPrec);
    end
end
   