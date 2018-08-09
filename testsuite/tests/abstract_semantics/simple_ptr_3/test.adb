procedure Test is
   type Point is record
      x : Integer;
      y : Integer;
   end record;
   x : access Integer;
   p : Point := (1, 2);
begin
   x := p.x'Access;
   x.all := 3;
end Ex1;
