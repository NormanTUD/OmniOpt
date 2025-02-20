my $x = $ARGV[0];
my $y = $ARGV[1];
my $z = $ARGV[2];

my $zahl = abs(log(abs(log(($x * $y + 1) / ($y + $x + 1)) * ($z + 1) + 1)));
print "RESULT: $zahl\n";
