 cat /usb/PMIs4 | shuf -n 15000 > 15krandImgs
 while read line; do file $line; done < 15krandImgs  | grep 603x400 > 15krandImgsRightSize
 vim 15krandImgsRightSize
 then clean
