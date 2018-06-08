procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
      z : Integer;
   end record;

   function F return Integer is
   begin
      return 42;
   end F;

   p : Point := (2, 3, 12);
begin
   p.x := F;
end Ex1;
