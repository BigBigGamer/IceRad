clear all
fid = fopen('names2017_01.txt','rt');      %file with needed track names
com = fopen('Jcomfile2017_01.txt','wt');      %file with directories of the needed files on the disk

while ~feof(fid)
    FileName = fgets(fid);
    UsefulData=FileName(24:53); %for alt
    YearMonth=UsefulData(1:6);
    YearMonthDay=UsefulData(1:8);
   % path='G:\Maria_Panfilova_hdf5_nasa\';
   % path='E:\Work\GitHub\IceRad\RAW_Data\';
    path = 'G:\DPR\';

%     s2=strcat(path,YearMonth,'\', YearMonthDay,strcat('\*',UsefulData,'*.hdf5'));
%     s22=strcat(path,YearMonth,'\', YearMonthDay);
    s2 = strcat(path,YearMonth,strcat('\*',UsefulData(3:8),UsefulData(11:14),'*.h5'));
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

