import sys
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

options = Options()
options.headless = True
browser = webdriver.Firefox(options=options, executable_path=r'test/geckodriver')

link = "https://imageseg.scads.ai/omnioptgui/?projectname=test_project&reservation=TESTRESERVATION&account=TESTACCOUNT&workdir=%2FWORKDIR%2F&param_0_name=randint&param_0_value_max=100&param_1_name=choice&param_1_values=1%2C2%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10&param_2_name=choiceint&param_2_value_min=1&param_2_value_max=10&param_3_name=uniform&param_3_value_min=1&param_3_value_max=2&param_4_name=quniform&param_4_value_min=1&param_4_value_max=10&param_4_value_q=3&param_5_name=loguniform&param_5_value_min=1&param_5_value_max=10&param_6_name=qloguniform&param_6_value_min=1&param_6_value_max=10&param_6_value_q=2&param_7_name=normal&param_7_mu=10&param_7_sigma=2&param_8_name=qnormal&mu_8=10&param_8_sigma=2&param_8_value_q=1&param_9_name=lognormal&param_9_mu=10&param_9_sigma=2&param_10_value_min=0&param_10_value_max=10&param_10_name=qlognormal&param_10_mu=10&param_10_sigma=2&param_10_value_q=1&param_11_name=&param_13_type=hp.randint&param_12_type=hp.randint&param_11_type=hp.randint&partition=ml&number_of_gpus=1&number_of_workers=5&mem_per_worker=2000&runtime=1&objective_program=perl%20-e%20%27print%20%22RESULT%3A%20%22.((%24x_0)%20%2B%20(%24x_1)%20%2B%20(%24x_2)%20%2B%20(%24x_3)%20%2B%20(%24x_4)%20%2B%20(%24x_5)%20%2B%20(%24x_6)%20%2B%20(%24x_7)%20%2B%20(%24x_8)%20%2B%20(%24x_9)%20%2B%20(%24x_10)).%22%5Cn%22%27&searchtype=tpe.suggest&max_evals=100&number_of_parameters=11&param_0_type=hp.randint&param_1_type=hp.choice&param_2_type=hp.choiceint&param_3_type=hp.uniform&param_4_type=hp.quniform&param_5_type=hp.loguniform&param_6_type=hp.qloguniform&param_7_type=hp.normal&param_8_type=hp.qnormal&param_9_type=hp.lognormal&param_10_type=hp.qlognormal"

browser.get(link)

bashcommand = browser.find_element_by_id("bashcommand")

should_be_text = 'curl https://imageseg.scads.ai/omnioptgui/omniopt_script.sh 2>/dev/null | bash -s -- --projectname=test_project --config_file=W0RBVEFdCm51bWJlcl9vZl93b3JrZXJzID0gNQpudW1fZ3B1c19wZXJfd29ya2VyID0gMQpwcmVjaXNpb24gPSA4CmFjY291bnQgPSBURVNUQUNDT1VOVApyZXNlcnZhdGlvbiA9IFRFU1RSRVNFUlZBVElPTgpwYXJ0aXRpb24gPSBtbApwcm9qZWN0bmFtZSA9IHRlc3RfcHJvamVjdAplbmFibGVfZ3B1cyA9IDEKbWVtX3Blcl9jcHUgPSAyMDAwCmNvbXB1dGluZ190aW1lID0gMQptYXhfZXZhbHMgPSAxMDAKYWxnb19uYW1lID0gdHBlLnN1Z2dlc3QKcmFuZ2VfZ2VuZXJhdG9yX25hbWUgPSBocC5yYW5kaW50Cm9iamVjdGl2ZV9wcm9ncmFtID0gcGVybCAtZSAncHJpbnQgIlJFU1VMVDogIi4oKCR4XzApICsgKCR4XzEpICsgKCR4XzIpICsgKCR4XzMpICsgKCR4XzQpICsgKCR4XzUpICsgKCR4XzYpICsgKCR4XzcpICsgKCR4XzgpICsgKCR4XzkpICsgKCR4XzEwKSkuIlxuIicKCltESU1FTlNJT05TXQpkaW1lbnNpb25zID0gMTEKCmRpbV8wX25hbWUgPSByYW5kaW50CnJhbmdlX2dlbmVyYXRvcl8wID0gaHAucmFuZGludAptYXhfZGltXzAgPSAxMDAKCmRpbV8xX25hbWUgPSBjaG9pY2UKcmFuZ2VfZ2VuZXJhdG9yXzEgPSBocC5jaG9pY2UKb3B0aW9uc18xID0gMSwyLDMsNCw1LDYsNyw4LDksMTAKCmRpbV8yX25hbWUgPSBjaG9pY2VpbnQKcmFuZ2VfZ2VuZXJhdG9yXzIgPSBocC5jaG9pY2UKb3B0aW9uc18yID0gMSwyLDMsNCw1LDYsNyw4LDksMTAKCmRpbV8zX25hbWUgPSB1bmlmb3JtCnJhbmdlX2dlbmVyYXRvcl8zID0gaHAudW5pZm9ybQptaW5fZGltXzMgPSAxCm1heF9kaW1fMyA9IDIKCmRpbV80X25hbWUgPSBxdW5pZm9ybQpyYW5nZV9nZW5lcmF0b3JfNCA9IGhwLnF1bmlmb3JtCm1pbl9kaW1fNCA9IDEKbWF4X2RpbV80ID0gMTAKcV80ID0gMwoKZGltXzVfbmFtZSA9IGxvZ3VuaWZvcm0KcmFuZ2VfZ2VuZXJhdG9yXzUgPSBocC5sb2d1bmlmb3JtCm1pbl9kaW1fNSA9IDEKbWF4X2RpbV81ID0gMTAKCmRpbV82X25hbWUgPSBxbG9ndW5pZm9ybQpyYW5nZV9nZW5lcmF0b3JfNiA9IGhwLnFsb2d1bmlmb3JtCm1pbl9kaW1fNiA9IDEKbWF4X2RpbV82ID0gMTAKcV82ID0gMgoKZGltXzdfbmFtZSA9IG5vcm1hbApyYW5nZV9nZW5lcmF0b3JfNyA9IGhwLm5vcm1hbAptdV83ID0gMTAKc2lnbWFfNyA9IDIKCmRpbV84X25hbWUgPSBxbm9ybWFsCnJhbmdlX2dlbmVyYXRvcl84ID0gaHAucW5vcm1hbAptdV84ID0gMTAKc2lnbWFfOCA9IDIKcV84ID0gMQoKZGltXzlfbmFtZSA9IGxvZ25vcm1hbApyYW5nZV9nZW5lcmF0b3JfOSA9IGhwLmxvZ25vcm1hbAptdV85ID0gMTAKc2lnbWFfOSA9IDIKCmRpbV8xMF9uYW1lID0gcWxvZ25vcm1hbApyYW5nZV9nZW5lcmF0b3JfMTAgPSBocC5xbG9nbm9ybWFsCm11XzEwID0gMTAKc2lnbWFfMTAgPSAyCnFfMTAgPSAxCgpbREVCVUddCmRlYnVnX3h0cmVtZSA9IDAKZGVidWcgPSAwCmluZm8gPSAwCndhcm5pbmcgPSAwCnN1Y2Nlc3MgPSAwCnN0YWNrID0gMApzaG93X2xpdmVfb3V0cHV0ID0gMApzYmF0Y2hfb3Jfc3J1biA9IHNiYXRjaApkZWJ1Z19zYmF0Y2hfc3J1biA9IDAKCltNT05HT0RCXQp3b3JrZXJfbGFzdF9qb2JfdGltZW91dCA9IDcyMDAKcG9sbF9pbnRlcnZhbCA9IDEwCmtpbGxfYWZ0ZXJfbl9ub19yZXN1bHRzID0gMTAwMDAwCg== --sbatch_command=c2JhdGNoIC1KIHRlc3RfcHJvamVjdCAgIC0tcmVzZXJ2YXRpb249VEVTVFJFU0VSVkFUSU9OICAgLUEgVEVTVEFDQ09VTlQgICAtLWNwdXMtcGVyLXRhc2s9NCAgIC0tZ3Jlcz1ncHU6NSAgIC0tZ3B1cy1wZXItdGFzaz0xICAgLS1udGFza3M9NSAgIC0tdGltZT0xOjAwOjAwICAgLS1tZW0tcGVyLWNwdT0yMDAwICAgLS1wYXJ0aXRpb249bWwgICBzYmF0Y2gucGwgLS1wcm9qZWN0PXRlc3RfcHJvamVjdCAgIC0tcGFydGl0aW9uPW1sICAtLW51bV9ncHVzX3Blcl93b3JrZXI9MSAgIC0tbWF4X3RpbWVfcGVyX3dvcmtlcj0xOjAwOjAw --workdir=/WORKDIR/ ; if [[ "$SHELL" == "/bin/bash" ]]; then history -r; elif [[ "$SHELL" == "/bin/zsh" ]]; then fc -R; fi'
is_text = bashcommand.text

exit_code = 1
if(is_text == should_be_text):
    print("OK")
    exit_code = 0
else:
    print("FAIL")

browser.quit()

sys.exit(exit_code)
