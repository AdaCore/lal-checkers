procedure Test(a: in out Integer; b: in out Integer) is
   tmp : Integer;
begin
   tmp := a;
   a := b;
   b := tmp;
end Ex1;
