extern crate cmake;

fn main() {
    let dst = cmake::build("gui");
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=neural-networks-gui");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=dylib=Qt5Core");
    println!("cargo:rustc-link-lib=dylib=Qt5Widgets");
    println!("cargo:rustc-link-lib=dylib=Qt5Gui");
}
