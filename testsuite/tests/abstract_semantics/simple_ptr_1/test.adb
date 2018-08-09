procedure Test is
   type Integer_Access is access all Integer;
   x : Integer_Access;
   y : Integer := 0;
   b : Boolean;
begin
   if b then
      x := y'Access;
   else
      x := null;
   end if;

   if x = null then
      y := 2;
   else
      y := x.all;
   end if;
end Ex1;
