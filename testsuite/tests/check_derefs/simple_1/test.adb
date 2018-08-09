procedure Test is
   type Integer_Access is access all Integer;
   x : Integer_Access := null;
   y : Integer := 0;
   b : Boolean;
begin
   if b then
      x := y'Access;
   end if;

   y := x.all;

   if b = True then
      y := 1;
   else
      y := -1;
   end if;
end Ex1;
