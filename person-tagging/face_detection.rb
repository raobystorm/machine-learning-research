detector = Mitene::Media::DlibFaceDetector.new(prefer_dnn: true)
results = {}
dir_path = '/home/ruizhong/nao_images/'
Dir.foreach(dir_path) do |img|
  next if img == '.' || img == '..' || img == '.DS_Store'
  pp img
  begin
  	results[dir_path + img] = detector.detect_one(dir_path + img)
  rescue => e
  	File.delete(dir_path + img)
  end
end

file = '/home/ruizhong/nao_results.txt'
File.open(file, 'a+') do |result_file|
  results.each do |img_file, detect_res|
    detect_res.each do |res|
      result_file.puts(img_file + ', x:' + res.face.x.to_s + ', y:' + res.face.y.to_s + ', width:' + res.face.width.to_s + ', height:' + res.face.height.to_s)
    end
    File.delete(img_file)
  end
end