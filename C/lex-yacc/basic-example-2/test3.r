fn test (a: i32, b:str ) -> i32 { 
	if (a==2) { 
println(b); let c = 5; c;
} else { println(b);
	let c = 6; c;
}
		a;
}

fn bigif(a: i32, b: i32, c:i32) -> i32 {
	if (a > b) {
	if (a > c) {a;} 
	else {c;}
	} else if (b > c) {b;} else {c;}
}


fn whilefn(a: i32, x: str) {
	let mut b = a;
	while (b > 2) {
		println(x);
		let b = b - 1;
	}
}

fn main ( ) {
	let mut a = 2 + 2 ; let a = test(5,"test") + a; println(test(2,"help"));
	whilefn(a,"again");
}

