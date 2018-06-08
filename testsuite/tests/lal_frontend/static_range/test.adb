procedure Ex1 is
   z : constant := 10 + 20;
   type My_Range is range 10 .. z;
   x : My_Range;
begin
   x := My_Range'First + My_Range'Last;
end Ex1;
