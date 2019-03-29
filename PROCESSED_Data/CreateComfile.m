clear all
fid = fopen('names2017_03.txt','rt');      %file with needed track names
com = fopen('comfile2017_03.txt','wt');      %file with directories of the needed files on the disk

while ~feof(fid)
    FileName = fgets(fid);
    UsefulData=FileName(23:53); %for alt
    YearMonth=strcat(UsefulData(1:4),'_',UsefulData(5:6));
    YearMonthDay=strcat(UsefulData(1:4),'_',UsefulData(5:6),'_',UsefulData(7:8));
   % path='G:\Maria_Panfilova_hdf5_nasa\';
   path='E:\Work\GitHub\IceRad\RAW_Data\';

%     s2=strcat(path,YearMonth,'\', YearMonthDay,strcat('\*',UsefulData,'*.hdf5'));
%     s22=strcat(path,YearMonth,'\', YearMonthDay);
    s2 = strcat(path,YearMonth,strcat('\*',UsefulData,'*.hdf5'));
    s22=strcat(path,YearMonth);

    MatchingFileNames = ls(s2);
    Amount=size(MatchingFileNames);
    %if yes, write down path to this file in comfile
        if Amount~=0
            for i=1:Amount(1)
                 fprintf(com,'%s\n', strcat(s22,'/',MatchingFileNames(i,:)));
            end
        end
end

fclose(fid);
fclose(com);

