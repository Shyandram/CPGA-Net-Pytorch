function [NIQE] = evaluation_niqe(test_folder)
%     test_folder = 'results\LAL_BDP-llie-L150';
    test_list = dir(test_folder);

    avg_niqe = 0;

    count = 0;

    for i=3:length(test_list)
        count = count + 1;

        % load images
        f1 = fullfile(test_folder,  test_list(i).name);
        filename =  test_list(i).name;
        if filename(end-1:end) == "db"
            continue;
        end
        img_test = imread(f1);

        % calculate metrics
        niqe_score = niqe(img_test);    
        
        avg_niqe = avg_niqe + niqe_score;


        % print
%         fprintf('%d\n', count);
%         fprintf('PSNR=%.4f\tSSIM=%.4f\n', peaksnr, ssimval);
        fprintf('\n%s NIQE=%.4f\n',test_list(i).name, niqe_score);
%         disp (f1);
%         disp (f2);
    end
    
    NIQE = avg_niqe/count;
	fprintf('Total\nNIQE=%.4f\n', NIQE);
    
end

