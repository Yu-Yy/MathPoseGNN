#ffmpeg -i /Extra/panzhiyu/CampusSeq1/demo_folder2/campus4-c0-%5d.png demo.mp4
# ffmpeg  -r 25 -pattern_type glob -i  -c:v libx264 out.mp4
ffmpeg  -y -r 25 -pattern_type glob -i '/Extra/panzhiyu/CampusSeq1/demo_folderv2/*.png' -c:v libx264 -vf fps=15 pose_demov2.mp4
#ffmpeg -i demo.mp4 -r 20 -f  campus4-c0-%5d.png