#email the files in $1...$n to emulation@pasqal.com
email_results(){
   attachments=${*/#/"-a "}
   msg="Hello Emulation Team!\n\nSee attachments for the emu-ct benchmark. \n\nDo not reply to this automatic email."
   subject='[no-reply] : Emu-CT weekly benchmark'
   echo -e $msg | mail -s "$subject" $attachments emulation@pasqal.com
}
