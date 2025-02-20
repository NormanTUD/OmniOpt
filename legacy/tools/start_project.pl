#!/usr/bin/perl

#sbatch -J 'Hist4D_ML_TEST_SQUARE' --mem-per-cpu='10000' --ntasks='8' --tasks-per-node=1 --gres=gpu:1 --time="12:00:00" --partition=ml sbatch.pl

my %options = (
        projectname => '',
        mempercpu => '10000',
        ntasks => '8',
        taskspernode => '1',
        gres => 'gpu:1',
        time => '12:00:00',
        partition => 'ml'
);

analyze_args(@ARGV);
main();

sub analyze_args {
        my @args = @_;

        foreach (@args) {
                if(m#^--projectname=(.*)#) {
                        $options{projectname} = $1;
                } elsif (m#^--mem-per-cpu=(\d+)$#) {
                        $options{mempercpu} = $1;
                } elsif (m#^--ntasks=(\d+)$#) {
                        $options{ntasks} = $1;
                } elsif (m#^--taskspernode=(\d+)$#) {
                        $options{taskspernode} = $1;
                } elsif (m#^--gres=(.*)$#) {
                        $options{gres} = $1;
                } elsif (m#^--time=(.*)$#) {
                        $options{time} = $1;
                } elsif (m#^--partition=(.*)$#) {
                        $options{partition} = $1;
                } else {
                        die "Unknown parameter $_!";
                }
        }
}

sub check_params {
        my $projectdir = "./projects/".$options{projectname}."/";
        
        if(!-d $projectdir) {
                die "Project dir $projectdir does not exist";
        } else {
                my $mongodb_lock = "./projects/Hist4D_ML_TEST/mongodb/mongodb.lock";
                if(-e $mongodb_lock) {
                        unlink $mongodb_lock or die $!;
                }
        }
        return 1;
}

sub main {
        if(check_params()) {
                my $command = qq#sbatch -J '$options{projectname}' --mem-per-cpu='$options{mempercpu}' --ntasks='$options{ntasks}' --tasks-per-node=$options{taskspernode} --gres=$options{gres} --time="$options{time}" --partition=$options{partition} sbatch.pl#;
                warn $command;
                system $command;
        } else {
                die "Invalid parameters";
        }
}
