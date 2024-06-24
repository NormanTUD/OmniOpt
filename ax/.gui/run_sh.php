<?php
	include("_header_base.php");
?>
    <link href="jquery-ui.css" rel="stylesheet">
    <style>
        body {
            font-family: Verdana, sans-serif;
        }
        .toc {
            margin-bottom: 20px;
        }
        .toc ul {
            list-style-type: none;
            padding: 0;
        }
        .toc li {
            margin-bottom: 5px;
        }
        .toc a {
            text-decoration: none;
            color: #007bff;
        }
        .toc a:hover {
            text-decoration: underline;
        }
    </style>
    <link href="prism.css" rel="stylesheet" />

    <h1>Create <tt>run.sh</tt>-file &amp; modify your program</h1>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#script-example">Script Example</a></li>
            <li><a href="#argument-parsing">Parse Arguments from the Command Line</a>
                <ul>
                    <li><a href="#sys-argv">Using sys.argv</a></li>
                    <li><a href="#argparse">Using argparse</a></li>
                </ul>
            </li>
	    <li><a href="#complex-example">Complex Example</a></li>
        </ul>
    </div>

    <h2 id="script-example">Script Example</h2>
    <p>To make your script robust enough for the environment of OmniOpt on HPC-Systems,
    it is recommended that you do not run your script directly in the objective program
    string. Rather, it is recommended that you create a <tt>run.sh</tt>-file from which
    your program gets run.</p>

    <p>It may look like this:</p>

    <pre><code class="language-bash">#!/bin/bash -l
# ^ Shebang-Line, so that it is known that this is a bash file
# -l means 'load this as login shell', so that /etc/profile gets loaded and you can use 'module load' or 'ml' as usual

# If you use this script not via `./run.sh' or just `srun run.sh', but like `srun bash run.sh', please add the '-l' there too.
# Like this:
# srun bash -l run.sh

# Load modules your program needs, always specify versions!
ml TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4 # Or whatever modules you need

# Load specific virtual environment (if applicable)
source /path/to/environment/bin/activate

# Load your script. $@ is all the parameters that are given to this run.sh file.
python3 /absolute/path/to_script.py $@
</code></pre>

    <p>Even though <tt>sbatch</tt> may inherit shell variables like loaded modules, 
    it is not recommended to rely on that heavily, because, especially when
    copying the <tt>curl</tt>-command from this website, you may forget loading
    the correct modules. This makes your script much more robust to changes.</p>

    <p>Also, always load specific module-versions and never let <tt>lmod</tt> guess
    the versions you want. Once these change, you'll almost certainly have problems
    otherwise.</p>

    <h2 id="argument-parsing">Parse Arguments from the Command Line</h2>

    <h3 id="sys-argv">Using sys.argv</h3>
    <p>The following Python program demonstrates how to parse command line arguments using <tt>sys.argv</tt>:</p>

    <pre><code class="language-python">import sys
epochs = int(sys.argv[1])
learning_rate = float(sys.argv[2])
model_name = sys.argv[3]

if epochs <= 0:
	print("Error: Number of epochs must be positive")
	sys.exit(1)
if not 0 < learning_rate < 1:
	print("Error: Learning rate must be between 0 and 1")
	sys.exit(2)
print(f"Running with epochs={epochs}, learning_rate={learning_rate}, model_name={model_name}")

# Your code here

# loss = model.fit(...)

loss = epochs + learning_rate

print(f"RESULT: {loss}")
</code></pre>

    <p>Example call:</p>
    <pre><code class="language-bash">python3 script.py 10 0.01 MyModel</code></pre>
    <p>Example OmniOpt-call:</p>
    <pre><code class="language-bash">python3 script.py %(epochs) %(learning_rate) %(model_name)</code></pre>

    <h3 id="argparse">Using argparse</h3>
    <p>The following Python program demonstrates how to parse command line arguments using <tt>argparse</tt>:</p>

    <pre><code class="language-python">import argparse
import sys

parser = argparse.ArgumentParser(description="Run a training script with specified parameters.")
parser.add_argument("epochs", type=int, help="Number of epochs")
parser.add_argument("learning_rate", type=float, help="Learning rate")
parser.add_argument("model_name", type=str, help="Name of the model")

args = parser.parse_args()

if args.epochs <= 0:
	print("Error: Number of epochs must be positive")
	sys.exit(1)
if not 0 < args.learning_rate < 1:
	print("Error: Learning rate must be between 0 and 1")
	sys.exit(2)

print(f"Running with epochs={args.epochs}, learning_rate={args.learning_rate}, model_name={args.model_name}")

# Your code here

# loss = model.fit(...)

loss = args.epochs + args.learning_rate

print(f"RESULT: {loss}")
</code></pre>

    <p>Example call:</p>
    <pre><code class="language-bash">python3 script.py --epochs 10 --learning_rate 0.01 --model_name MyModel</code></pre>
    <p>Example OmniOpt-call:</p>
    <pre><code class="language-bash">python3 script.py --epochs %(epochs) --learning_rate %(learning_rate) --model_name %(model_name)</code></pre>

    <p><strong>Advantages of using <tt>argparse</tt>:</strong></p>
    <ul>
        <li>Order of arguments does not matter; they are matched by name.</li>
        <li>Type checking is automatically handled based on the type specified in <tt>add_argument</tt>.</li>
        <li>Generates helpful usage messages if the arguments are incorrect or missing.</li>
        <li>Supports optional arguments and more complex argument parsing needs.</li>
    </ul>

    <h2 id="complex-example">Complex example</h2>

    <p>The following program passes hyperparameters to YOLOv5 and  parses it's output for the last loss, which is the printed with the required RESULT-string:</p>

    <pre><code class="language-bash">
#!/bin/bash -l

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR

ml modenv/hiera GCCcore/11.3.0 Python/3.9.6

if [[ ! -e ~/.alpha_yoloenv/bin/activate ]]; then
	python3 -mvenv ~/.alpha_yoloenv/
	source ~/.alpha_yoloenv/bin/activate
	pip3 install -r requirements.txt
fi

source ~/.alpha_yoloenv/bin/activate



function echoerr() {
	echo "$@" 1>&2
}

function red_text {
	echoerr -e "\[31m$1\[0m"
}

set -e
set -o pipefail
set -u

function calltracer () {
	echo 'Last file/last line:'
	caller
}
trap 'calltracer' ERR

function help () {
	echo "Possible options:"
	echo "  --batchsize=INT                                    default value: 130"
	echo "  --epochs=INT                                       default value: 1500"
	echo "  --img=INT                                          default value: 512"
	echo "  --patience=INT                                     default value: 200"
	echo "	--lr0=FLOAT                                        default value: 0.01"
	echo "	--lrf=FLOAT                                        default value: 0.1"
	echo "	--momentum=FLOAT                                   default value: 0.937"
	echo "	--weight_decay=FLOAT                               default value: 0.0005"
	echo "	--warmup_epochs=FLOAT                              default value: 3.0"
	echo "	--warmup_momentum=FLOAT                            default value: 0.8"
	echo "	--warmup_bias_lr=FLOAT                             default value: 0.1"
	echo "	--box=FLOAT                                        default value: 0.05"
	echo "	--cls=FLOAT                                        default value: 0.3"
	echo "	--cls_pw=FLOAT                                     default value: 1.0"
	echo "	--obj=FLOAT                                        default value: 0.7"
	echo "	--obj_pw=FLOAT                                     default value: 1.0"
	echo "	--iou_t=FLOAT                                      default value: 0.20"
	echo "	--anchor_t=FLOAT                                   default value: 4.0"
	echo "	--fl_gamma=FLOAT                                   default value: 0.0"
	echo "	--hsv_h=FLOAT                                      default value: 0.015"
	echo "	--hsv_s=FLOAT                                      default value: 0.7"
	echo "	--hsv_v=FLOAT                                      default value: 0.4"
	echo "	--degrees=FLOAT                                    default value: 360"
	echo "	--translate=FLOAT                                  default value: 0.1"
	echo "	--scale=FLOAT                                      default value: 0.9"
	echo "	--shear=FLOAT                                      default value: 0.0"
	echo "	--perspective=FLOAT                                default value: 0.001"
	echo "	--flipud=FLOAT                                     default value: 0.3"
	echo "	--fliplr=FLOAT                                     default value: 0.5"
	echo "	--mosaic=FLOAT                                     default value: 1.0"
	echo "	--mixup=FLOAT                                      default value: 0.3"
	echo "	--copy_paste=FLOAT                                 default value: 0.4"
	echo "  --model"
	echo "  --help                                             this help"
	echo "  --debug                                            Enables debug mode (set -x)"

	exit $1
}

export batchsize=130
export epochs=1500
export img=512
export patience=200
export model=yolov5s.yaml
export img=512
export patience=200
export lr0=0.01
export lrf=0.1
export momentum=0.937
export weight_decay=0.0005
export warmup_epochs=3.0
export warmup_momentum=0.8
export warmup_bias_lr=0.1
export box=0.05
export cls=0.3
export cls_pw=1.0
export obj=0.7
export obj_pw=1.0
export iou_t=0.20
export anchor_t=4.0
export fl_gamma=0.0
export hsv_h=0.015
export hsv_s=0.7
export hsv_v=0.4
export degrees=360
export translate=0.1
export scale=0.9
export shear=0.0
export perspective=0.001
export flipud=0.3
export fliplr=0.5
export mosaic=1.0
export mixup=0.3
export copy_paste=0.4

for i in $@; do
case $i in
	--batchsize=*)
		batchsize="${i#*=}"
		re='^[+-]?[0-9]+$'
		if ! [[ $batchsize =~ $re ]] ; then
			red_text "error: Not a INT: $i" >&2
			help 1
		fi
		shift
		;;
	--epochs=*)
		epochs="${i#*=}"
		re='^[+-]?[0-9]+$'
		if ! [[ $epochs =~ $re ]] ; then
			red_text "error: Not a INT: $i" >&2
			help 1
		fi
		shift
		;;
	--img=*)
		img="${i#*=}"
		re='^[+-]?[0-9]+$'
		if ! [[ $img =~ $re ]] ; then
			red_text "error: Not a INT: $i" >&2
			help 1
		fi
		shift
		;;
	--patience=*)
		patience="${i#*=}"
		re='^[+-]?[0-9]+$'
		if ! [[ $patience =~ $re ]] ; then
			red_text "error: Not a INT: $i" >&2
			help 1
		fi
		shift
		;;
	--model=*)
		model="${i#*=}"
		shift
		;;
	--lr0=*)
		lr0="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $lr0 =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--lrf=*)
		lrf="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $lrf =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--momentum=*)
		momentum="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $momentum =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--weight_decay=*)
		weight_decay="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $weight_decay =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--warmup_epochs=*)
		warmup_epochs="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $warmup_epochs =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--warmup_momentum=*)
		warmup_momentum="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $warmup_momentum =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--warmup_bias_lr=*)
		warmup_bias_lr="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $warmup_bias_lr =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--box=*)
		box="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $box =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--cls=*)
		cls="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $cls =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--cls_pw=*)
		cls_pw="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $cls_pw =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--obj=*)
		obj="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $obj =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--obj_pw=*)
		obj_pw="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $obj_pw =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--iou_t=*)
		iou_t="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $iou_t =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--anchor_t=*)
		anchor_t="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $anchor_t =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--fl_gamma=*)
		fl_gamma="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $fl_gamma =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--hsv_h=*)
		hsv_h="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $hsv_h =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--hsv_s=*)
		hsv_s="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $hsv_s =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--hsv_v=*)
		hsv_v="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $hsv_v =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--degrees=*)
		degrees="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $degrees =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--translate=*)
		translate="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $translate =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--scale=*)
		scale="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $scale =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--shear=*)
		shear="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $shear =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--perspective=*)
		perspective="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $perspective =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--flipud=*)
		flipud="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $flipud =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--fliplr=*)
		fliplr="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $fliplr =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--mosaic=*)
		mosaic="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $mosaic =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--mixup=*)
		mixup="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $mixup =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	--copy_paste=*)
		copy_paste="${i#*=}"
		re="^[+-]?[0-9]+([.][0-9]+)?$"
		if ! [[ $copy_paste =~ $re ]] ; then
			red_text "error: Not a FLOAT: $i" >&2
			help 1
		fi
		shift
		;;
	-h|--help)
		help 0
		;;
	--debug)
		set -x
		;;
	*)
		red_text "Unknown parameter $i" >&2
		help 1
		;;
esac
done

run_uuid=$(uuidgen)

hyps_file=$SCRIPT_DIR/data/hyps/hyperparam_${run_uuid}.yaml

hyperparams_file_contents="
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Hyperparameters for high-augmentation COCO training from scratch
# python train.py --batch 32 --cfg yolov5m6.yaml --weights "" --data coco.yaml --img 1280 --epochs 300
# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials

lr0: $lr0 # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: $lrf # final OneCycleLR learning rate (lr0 * lrf)
momentum: $momentum # SGD momentum/Adam beta1
weight_decay: $weight_decay # optimizer weight decay 5e-4
warmup_epochs: $warmup_epochs # warmup epochs (fractions ok)
warmup_momentum: $warmup_momentum # warmup initial momentum
warmup_bias_lr: $warmup_bias_lr # warmup initial bias lr
box: $box # box loss gain
cls: $cls # cls loss gain
cls_pw: $cls_pw # cls BCELoss positive_weight
obj: $obj # obj loss gain (scale with pixels)
obj_pw: $obj_pw # obj BCELoss positive_weight
iou_t: $iou_t # IoU training threshold
anchor_t: $anchor_t # anchor-multiple threshold
# anchors: $# anchors # anchors per output layer (0 to ignore)
fl_gamma: $fl_gamma # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: $hsv_h # image HSV-Hue augmentation (fraction)
hsv_s: $hsv_s # image HSV-Saturation augmentation (fraction)
hsv_v: $hsv_v # image HSV-Value augmentation (fraction)
degrees: $degrees # image rotation (+/- deg)
translate: $translate # image translation (+/- fraction)
scale: $scale # image scale (+/- gain)
shear: $shear # image shear (+/- deg)
perspective: $perspective # image perspective (+/- fraction), range 0-0.001
flipud: $flipud # image flip up-down (probability)
fliplr: $fliplr # image flip left-right (probability)
mosaic: $mosaic # image mosaic (probability)
mixup: $mixup # image mixup (probability)
copy_paste: $copy_paste # segment copy-paste (probability)
"

echo "$hyperparams_file_contents" > "$hyps_file"

python3 $SCRIPT_DIR/train.py --cfg "$model" --multi-scale --batch $batchsize --data $SCRIPT_DIR/data/dataset.yaml --epochs $epochs --cache --img $img --hyp "$hyps_file" --patience $patience 2>&1 \
| awk '{print;print > "/dev/stderr"}' \
| egrep '[0-9]G' \
| egrep '[0-9]/[0-9]' \
| grep -v Class \
| sed -e 's/.*G\s*//' \
| egrep '^[0-9]+\.[0-9]+' \
| tail -n1 \
| sed -e 's/\s*[0-9]*\s*[0-9]*:.*//' \
| sed -e 's#\s\s*#\n#g' \
| perl -e '$i = 1; while (<>) { print qq#RESULT$i: $_#; $i++; }'

</code></pre>
    

    <script src="prism.js"></script>
    <script>
        Prism.highlightAll();
    </script>
</body>
</html>

