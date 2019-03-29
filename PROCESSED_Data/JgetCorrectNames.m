clear all
fid=fopen('nasa2017.txt','rt');
fileName=fgets(fid);
fileName=fgets(fid);
fileName=fgets(fid);
fileName=fgets(fid);
com=fopen('names2017.txt','wt');

fname_starter='2A.';

while ~feof(fid)
    fileName=fgets(fid);
    %fprintf(com,'%s',fileName(107:end));
    if length(fileName)>10
        fname_start=strfind(fileName,fname_starter);
        fprintf(com,'%s',fileName(fname_start(end):end));
        %fprintf(com,'%s\n',strcat(fileName(105:115),fileName(134:end)));
    end
end
fclose(fid);
fclose(com);
