function [W,bias,neurons,input] = readTF(fileName)

%fileName = 'mnist_tanh_3_5.txt';
fileID = fopen(fileName,'r');
formatSpec = '%s';
C = textscan(fileID,formatSpec);
Matrix = 0;
layer = 0;
dim = length(C{1});
for i=1:dim
    if strcmp(C{1}{i},'Tanh')
        layer = layer+1;
        clear Matrix
    elseif contains(C{1}{i},'[[')
        startMatrix = 1;
        Matrix(1,1) = str2num(C{1}{i}(3:end-1));
        Nrow = 1;
        Ncol = 2;
    elseif contains(C{1}{i},']]')
         Matrix(Nrow,Ncol) = str2num(C{1}{i}(1:end-2));
         startMatrix = 0;
         if layer==1
             input = Ncol;
         end
         neurons(layer) = Nrow;
         W{layer,1} = Matrix;
         
         
    elseif contains(C{1}{i},']')
        Matrix(Nrow,Ncol) = str2num(C{1}{i}(1:end-2));
        if startMatrix
            Nrow = Nrow + 1;
            Ncol = 1;
        else
            neurons(layer) = Ncol;
            bias{layer,1} = Matrix';
        end
    elseif contains(C{1}{i},'[')
        if startMatrix
            Matrix(Nrow,Ncol) = str2num(C{1}{i}(2:end));
            Ncol = Ncol + 1;
        else
            clear Matrix
            Matrix(1,1) = str2num(C{1}{i}(2:end));
            Nrow = 1;
            Ncol = 2;
        end
    else
        Matrix(Nrow,Ncol) = str2num(C{1}{i});
        Ncol = Ncol + 1;
    end
end
        