use strict;
use warnings;
use Data::Dumper;
use Term::ANSIColor;

my %options = (
        jobid => undef,
        command => [],
        repeat => undef,
	clear => 0
);

analyze_args(@ARGV);

main();

sub main {
        if($options{jobid}) {
                if(@{$options{command}}) {
			if($options{clear}) {
				system("clear");
			}
                        my @nodes = get_nodelist();
                        my $do = 1;
                        while ($do) {
                                run_command_on_servers(@nodes);
                                if(!defined($options{repeat})) {
                                        $do = 0;
                                } else {
                                        sleep $options{repeat};
					if($options{clear}) {
						system("clear");
					}
                                }
                        }
                } else {
                        die "No --command=... given";
                }
        } else {
                die "No --jobid=... given";
        }
}

sub get_nodelist {
        my $command = qq#squeue --format="%N" -j $options{jobid} | tail -n1#;

        #my $output = 'taurusml[10,2-4,8-10,20-25]'; # qx($command);
        my $output = qx($command);

        my @servers;
        if($output =~ m#^(.*)\[([0-9,\-]+)\]$#) {
                my $servername = $1;
                my $serverlist = $2;

                #die $serverlist;
                foreach my $item (split /,/, $serverlist) {
                        if($item =~ m#(\d+)-(\d+)#) {
                                foreach my $this_item ($1 .. $2) {
                                        push @servers, "$servername$this_item";
                                }
                        } else {
                                push @servers, "$servername$item";
                        }
                }
        } elsif($output =~ m#^(.*)([0-9]+)$#) {
                push @servers, $output;
        } else {
                die "Invalid output: $output";
        }
        return uniq(sort { get_index_from_servername($a) <=> get_index_from_servername($b) || $a cmp $b } @servers);
}

sub _help {
        my $exit_code = shift // 0;


        print <<EOF;
Give this script a job id of a running job and it will run a command on every node.

--help                                  This help
--jobid=12345                           Get nodes of the currently running job 12345
--command="ps auxf"                     Run the command "ps auxf" (can be added multiple times to add many commands)
--repeat=10                             Repeat every 10 seconds

Examples:

Show job 12345's nodes "ps auxf" every 10 seconds:

> perl run_on_every_node.pl --jobid=12345 --command="ps auxf" --repeat=10

Show all the open file descriptors of a certain job whose script is called train2

> perl run_on_every_node.pl --jobid=12345 --command="ls -aslrt1 /proc/\\\$(ps auxf | grep train2 | grep -v grep | awk '{ print \\\$2 }')/fd"

EOF

        exit $exit_code;
}

sub analyze_args {
        foreach (@_) {
                if(/^--jobid=(\d+)$/) {
                        $options{jobid} = $1;
                } elsif(/^--command=(.*)$/) {
                        push @{$options{command}}, $1;
                } elsif(/^--repeat=(\d*)$/) {
                        $options{repeat} = $1;
                } elsif(/^--clear$/) {
                        $options{clear} = 1;
                } elsif(/^--help$/) {
			_help(0);
                } else {
                        warn "Unknown parameter $_";
			_help(1);
                }
        }
}

sub uniq {
        my %seen;
        grep !$seen{$_}++, @_;
}

sub run_command_on_servers {
        my @servers = @_;

        foreach my $server (@servers) {
                print color("red").$server.color("reset").":\n";
		foreach my $command (@{$options{command}}) {
			my $ssh_command = qq#ssh $server "$command"#;
			print qx($ssh_command);
		}
        }
}


sub get_index_from_servername {
	my $servername = shift;

	if($servername =~ m#.*?(\d+)#) {
		return $1;
	}

	return $servername;
}
